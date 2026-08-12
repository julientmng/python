import os
import re
import tempfile
from pathlib import Path

import streamlit as st


st.set_page_config(
    page_title="EyeSuite ESBX → ESX Converter",
    page_icon="👁️",
    layout="centered",
)

st.title("👁️ EyeSuite ESBX → ESX Converter")
st.caption("Convert EyeSuite patient backup files (.esbx) into export files (.esx).")

st.info(
    "The converter works as a streaming byte-level transformation. "
    "It is designed for very large EyeSuite patient files and does not load "
    "the complete file into RAM."
)


def inspect_esbx(path: Path):
    """Inspect the beginning/stream of the XML for EyeSuite markers."""
    root_found = False
    context = None
    database_id = None

    # We only need to inspect text markers. UTF-8 is expected for EyeSuite XML.
    # Read in chunks to avoid loading a huge file.
    with path.open("rb") as f:
        data = b""
        while len(data) < 8 * 1024 * 1024:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            data += chunk

            if b"<HSEyeSuite" in data or b"<HSEyeSuite " in data:
                root_found = True

            if context is None:
                m = re.search(
                    rb"<CREATION_CONTEXT>\s*([^<]+?)\s*</CREATION_CONTEXT>",
                    data,
                    re.I,
                )
                if m:
                    context = m.group(1).decode("utf-8", errors="replace").strip()

            if database_id is None:
                m = re.search(
                    rb"<DatabaseId>\s*([^<]+?)\s*</DatabaseId>",
                    data,
                    re.I,
                )
                if m:
                    database_id = m.group(1).decode("utf-8", errors="replace").strip()

            if root_found and context is not None:
                break

    return root_found, context, database_id


def convert_esbx(source: Path, destination: Path):
    """Convert Backup -> Export while preserving all other file bytes."""
    old_variants = [
        b"<CREATION_CONTEXT>Backup</CREATION_CONTEXT>",
        b"<CREATION_CONTEXT>BACKUP</CREATION_CONTEXT>",
        b"<CREATION_CONTEXT>backup</CREATION_CONTEXT>",
    ]
    new_value = b"<CREATION_CONTEXT>Export</CREATION_CONTEXT>"

    # Keep enough bytes between chunks so the XML marker cannot be split.
    overlap = max(len(x) for x in old_variants) - 1
    replaced = 0
    total = source.stat().st_size
    processed = 0

    with source.open("rb") as src, destination.open("wb") as dst:
        buffer = b""

        while True:
            chunk = src.read(4 * 1024 * 1024)

            if not chunk:
                break

            buffer += chunk

            if len(buffer) > overlap:
                write_part = buffer[:-overlap]
                buffer = buffer[-overlap:]

                for old in old_variants:
                    count = write_part.count(old)
                    if count:
                        write_part = write_part.replace(old, new_value)
                        replaced += count

                dst.write(write_part)
                processed += len(write_part)

        for old in old_variants:
            count = buffer.count(old)
            if count:
                buffer = buffer.replace(old, new_value)
                replaced += count

        dst.write(buffer)
        processed += len(buffer)

    if replaced == 0:
        destination.unlink(missing_ok=True)
        raise ValueError(
            "Could not find <CREATION_CONTEXT>Backup</CREATION_CONTEXT>. "
            "The file may already be an ESX export or use an unsupported format."
        )

    return replaced, total


def validate_output(path: Path):
    root_found, context, database_id = inspect_esbx(path)

    if not root_found:
        raise ValueError("Output does not appear to be an EyeSuite XML file.")

    if not context or context.upper() != "EXPORT":
        raise ValueError(
            f"Output validation failed. Creation context is {context!r}, "
            "expected 'Export'."
        )

    return database_id


def human_size(size):
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:,.1f} {unit}"
        value /= 1024


uploaded = st.file_uploader(
    "Upload an EyeSuite patient backup",
    type=["esbx"],
    help="Select a .ESBX patient file exported/backed up by EyeSuite.",
)

if uploaded:
    original_name = Path(uploaded.name).name
    source_suffix = Path(original_name).suffix.lower()

    st.subheader("File")
    st.write(f"**{original_name}**")
    st.write(f"Size: **{human_size(uploaded.size)}**")

    if source_suffix != ".esbx":
        st.error("Please upload an .ESBX file.")
        st.stop()

    # Stream the uploaded Streamlit file to disk rather than keeping another
    # large in-memory copy.
    temp_dir = Path(tempfile.mkdtemp(prefix="eyesuite_converter_"))
    source = temp_dir / original_name
    destination = temp_dir / (Path(original_name).stem + ".esx")

    try:
        with source.open("wb") as f:
            while True:
                chunk = uploaded.read(8 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

        root_found, context, database_id = inspect_esbx(source)

        st.subheader("EyeSuite information")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("EyeSuite XML", "Detected" if root_found else "Not detected")
        with col2:
            st.metric("Creation context", context or "Unknown")

        if database_id:
            st.caption(f"Database ID: `{database_id}`")

        if not root_found:
            st.error(
                "This file does not appear to contain the expected EyeSuite XML structure."
            )
            st.stop()

        if not context or context.upper() != "BACKUP":
            st.warning(
                f"The file's creation context is {context!r}, not 'Backup'. "
                "Conversion was not performed."
            )
            st.stop()

        if st.button("🔄 Convert to ESX", type="primary", use_container_width=True):
            progress = st.progress(0, text="Converting…")

            try:
                replaced, total = convert_esbx(source, destination)
                progress.progress(90, text="Validating converted file…")

                output_database_id = validate_output(destination)

                # Verify that the database identifier was preserved.
                if database_id and output_database_id and database_id != output_database_id:
                    raise ValueError(
                        "Validation failed: DatabaseId changed during conversion."
                    )

                progress.progress(100, text="Conversion complete.")

                st.success(
                    f"Conversion successful — changed {replaced} EyeSuite "
                    "creation-context marker."
                )

                st.subheader("Converted file")
                st.write(f"**{destination.name}**")
                st.write(f"Size: **{human_size(destination.stat().st_size)}**")

                # Streamlit reads this file only when the user clicks download.
                with destination.open("rb") as f:
                    file_bytes = f.read()

                st.download_button(
                    "⬇️ Download converted ESX",
                    data=file_bytes,
                    file_name=destination.name,
                    mime="application/xml",
                    use_container_width=True,
                )

                st.caption(
                    "The source ESBX is not modified. The converted ESX is generated "
                    "as a separate file."
                )

            except Exception as exc:
                progress.empty()
                st.error(f"Conversion failed: {exc}")

    finally:
        # Do not delete files before Streamlit has finished rendering the
        # download button. Streamlit reruns the script when download is used,
        # so cleanup is intentionally omitted for this request.
        pass

st.divider()
st.caption(
    "EyeSuite ESBX → ESX Converter • Test converted files in a non-production "
    "EyeSuite environment before importing patient data."
)
