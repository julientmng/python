#!/usr/bin/env python3
"""
EyeSuite ESBX -> ESX converter

Important:
- ESBX and ESX are EyeSuite XML-based formats.
- This tool does NOT simply rename the extension.
- It preserves the XML/data and changes the backup/export context.
- It performs validation before writing the output.
- It streams the XML transformation to avoid loading a 300+ MB file into RAM.

Usage:
    python esbx_to_esx_converter.py patient.esbx
    python esbx_to_esx_converter.py patient.esbx -o patient.esx
    python esbx_to_esx_converter.py *.esbx --output-dir converted

By default, output is written beside the source file with .esx extension.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET


def local_name(tag: str) -> str:
    """Return XML local name, ignoring a possible namespace."""
    return tag.rsplit("}", 1)[-1]


def inspect_root(path: Path) -> tuple[str, str, str]:
    """
    Read only enough of the XML to identify the root and creation context.
    Returns (root_name, context, database_id).
    """
    context = ""
    database_id = ""
    root_name = ""

    for event, elem in ET.iterparse(path, events=("start", "end")):
        if event == "start":
            name = local_name(elem.tag)

            if not root_name:
                root_name = name

            if name == "CREATION_CONTEXT":
                # Value is available on end event.
                pass

        else:
            name = local_name(elem.tag)

            if name == "CREATION_CONTEXT" and not context:
                context = (elem.text or "").strip()

            elif name == "DatabaseId" and not database_id:
                database_id = (elem.text or "").strip()

            if context and database_id and root_name:
                break

    return root_name, context, database_id


def transform_esbx_to_esx(source: Path, destination: Path) -> dict:
    """
    Transform an ESBX backup XML into an ESX export XML.

    The transformation deliberately limits itself to the confirmed
    structural distinction: CREATION_CONTEXT Backup -> Export.

    The original XML is copied through an incremental XML serializer.
    This avoids loading the 300+ MB patient file into memory.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    root_name, context, database_id = inspect_root(source)

    if root_name != "HSEyeSuite":
        raise ValueError(
            f"Unexpected EyeSuite root: {root_name!r}. "
            "This does not appear to be a normal EyeSuite XML file."
        )

    if context.upper() != "BACKUP":
        raise ValueError(
            f"Expected <CREATION_CONTEXT>Backup</CREATION_CONTEXT>, "
            f"but found {context!r}. Refusing to modify the file."
        )

    # Write to a temporary file in the destination directory so a failed
    # conversion never leaves a partially written .esx as the final file.
    fd, temp_name = tempfile.mkstemp(
        prefix=destination.stem + "_",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    os.close(fd)
    temp_path = Path(temp_name)

    changed = False
    patient_count = 0
    exam_count = 0

    try:
        # iterparse gives us a streaming parser.
        context_iter = ET.iterparse(source, events=("start", "end"))

        # We cannot safely use ElementTree.write() event-by-event, because
        # namespaces and XML declarations may be altered. Instead, collect
        # the root and write incrementally by serializing completed elements.
        #
        # For EyeSuite files, the file has one large root document. A
        # full-tree parse would consume too much RAM, so we use a SAX-style
        # approach below.
        import xml.sax
        from xml.sax.saxutils import XMLGenerator

        class Handler(xml.sax.ContentHandler):
            def __init__(self, out):
                super().__init__()
                self.out = XMLGenerator(out, encoding="utf-8")
                self.context_depth = 0
                self.in_context = False
                self.current_context_text = []
                self.root_seen = False

            def startDocument(self):
                self.out.startDocument()

            def endDocument(self):
                self.out.endDocument()

            def startPrefixMapping(self, prefix, uri):
                self.out.startPrefixMapping(prefix, uri)

            def endPrefixMapping(self, prefix):
                self.out.endPrefixMapping(prefix)

            def startElement(self, name, attrs):
                nonlocal_vars = None
                if name == "CREATION_CONTEXT":
                    self.in_context = True
                    self.current_context_text = []
                self.out.startElement(name, attrs)

            def characters(self, content):
                if self.in_context:
                    self.current_context_text.append(content)
                self.out.characters(content)

            def endElement(self, name):
                nonlocal changed
                if name == "CREATION_CONTEXT" and self.in_context:
                    value = "".join(self.current_context_text).strip()
                    if value.upper() == "BACKUP":
                        # XMLGenerator has already written Backup. We need
                        # to replace it, which cannot be done retroactively.
                        # This handler therefore isn't used for the actual
                        # transformation.
                        pass
                    self.in_context = False
                self.out.endElement(name)

        # Use a lexical byte-level replacement instead. This is safer for a
        # huge EyeSuite XML because the target element has no ambiguity and
        # preserves all other bytes, formatting, encodings and embedded data.
        with open(source, "rb") as src, open(temp_path, "wb") as dst:
            old = b"<CREATION_CONTEXT>Backup</CREATION_CONTEXT>"
            new = b"<CREATION_CONTEXT>Export</CREATION_CONTEXT>"

            # Also handle the common capitalization variants.
            old_variants = [
                b"<CREATION_CONTEXT>Backup</CREATION_CONTEXT>",
                b"<CREATION_CONTEXT>BACKUP</CREATION_CONTEXT>",
                b"<CREATION_CONTEXT>backup</CREATION_CONTEXT>",
            ]

            overlap = max(len(x) for x in old_variants) - 1
            buffer = b""

            while True:
                chunk = src.read(4 * 1024 * 1024)
                if not chunk:
                    break

                buffer += chunk

                if len(buffer) > overlap:
                    write_part = buffer[:-overlap]
                    buffer = buffer[-overlap:]

                    for old_value in old_variants:
                        if old_value in write_part:
                            write_part = write_part.replace(old_value, new)
                            changed = True

                    dst.write(write_part)

            for old_value in old_variants:
                if old_value in buffer:
                    buffer = buffer.replace(old_value, new)
                    changed = True

            dst.write(buffer)

        if not changed:
            raise ValueError(
                "No <CREATION_CONTEXT>Backup</CREATION_CONTEXT> element was found. "
                "The file was not changed."
            )

        # Basic post-conversion checks.
        new_root, new_context, new_database_id = inspect_root(temp_path)

        if new_root != "HSEyeSuite":
            raise ValueError("Validation failed: invalid EyeSuite root.")

        if new_context.upper() != "EXPORT":
            raise ValueError(
                f"Validation failed: output context is {new_context!r}, "
                "not Export."
            )

        if database_id and new_database_id and database_id != new_database_id:
            raise ValueError("Validation failed: DatabaseId changed.")

        # Atomic final move.
        os.replace(temp_path, destination)

        return {
            "source": str(source),
            "destination": str(destination),
            "root": new_root,
            "creation_context": new_context,
            "database_id": new_database_id,
            "changed": changed,
            "size": destination.stat().st_size,
        }

    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert EyeSuite .esbx patient backup XML to .esx export XML."
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="One or more .esbx files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output .esx filename (only valid when converting one input)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for converted files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .esx files",
    )

    args = parser.parse_args()

    if args.output and len(args.files) != 1:
        parser.error("--output can only be used with one input file.")

    if args.output and args.output.suffix.lower() != ".esx":
        parser.error("--output must have a .esx extension.")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    failures = 0

    for source in args.files:
        if not source.exists():
            print(f"[ERROR] Not found: {source}", file=sys.stderr)
            failures += 1
            continue

        if source.suffix.lower() != ".esbx":
            print(f"[ERROR] Not an .esbx file: {source}", file=sys.stderr)
            failures += 1
            continue

        if args.output:
            destination = args.output
        elif args.output_dir:
            destination = args.output_dir / (source.stem + ".esx")
        else:
            destination = source.with_suffix(".esx")

        if destination.exists() and not args.force:
            print(
                f"[ERROR] Output already exists: {destination}\n"
                "        Use --force to overwrite.",
                file=sys.stderr,
            )
            failures += 1
            continue

        print(f"Converting: {source}")
        print(f"       to: {destination}")

        try:
            result = transform_esbx_to_esx(source, destination)
            print("[OK] Conversion completed.")
            print(f"     Size: {result['size']:,} bytes")
            print(f"     Context: {result['creation_context']}")
            if result["database_id"]:
                print(f"     DatabaseId: {result['database_id']}")
        except Exception as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
