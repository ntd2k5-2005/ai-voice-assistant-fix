import stat
import zipfile

import pytest

from rag.config import ParserConfig
from rag.ingestion.parsers import ParserError, ParserErrorCode, validate_zip_archive


def _archive(path, members, *, compression=zipfile.ZIP_DEFLATED):
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for name, payload in members:
            archive.writestr(name, payload)
    return path


@pytest.mark.parametrize("unsafe_name", ["../escape.xml", "/absolute.xml", "C:/drive.xml", "%2e%2e/encoded.xml"])
def test_zip_preflight_rejects_traversal_and_absolute_members(tmp_path, unsafe_name):
    path = _archive(tmp_path / "unsafe.docx", [(unsafe_name, "content")])
    with pytest.raises(ParserError) as error:
        validate_zip_archive(path, ParserConfig())
    assert error.value.code is ParserErrorCode.CORRUPT_CONTENT
    assert error.value.stage == "container_preflight"


def test_zip_preflight_rejects_member_count_size_total_and_ratio(tmp_path):
    count_path = _archive(tmp_path / "count.docx", [(f"{i}.xml", "x") for i in range(3)])
    with pytest.raises(ParserError):
        validate_zip_archive(count_path, ParserConfig(max_archive_members=2))

    member_path = _archive(tmp_path / "member.docx", [("large.xml", "x" * 100)])
    with pytest.raises(ParserError):
        validate_zip_archive(
            member_path,
            ParserConfig(
                max_archive_member_bytes=50,
                max_archive_uncompressed_bytes=200,
            ),
        )
    total_path = _archive(
        tmp_path / "total.docx", [("one.xml", "x" * 40), ("two.xml", "y" * 40)]
    )
    with pytest.raises(ParserError):
        validate_zip_archive(
            total_path,
            ParserConfig(
                max_archive_member_bytes=50,
                max_archive_uncompressed_bytes=60,
            ),
        )
    with pytest.raises(ParserError) as ratio_error:
        validate_zip_archive(
            member_path,
            ParserConfig(max_archive_compression_ratio=2.0),
        )
    assert "compression ratio" in str(ratio_error.value)


def test_zip_preflight_rejects_symlinks_and_encrypted_flags(tmp_path):
    symlink_path = tmp_path / "symlink.docx"
    with zipfile.ZipFile(symlink_path, "w") as archive:
        info = zipfile.ZipInfo("word/link.xml")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, "word/document.xml")
    with pytest.raises(ParserError) as symlink_error:
        validate_zip_archive(symlink_path, ParserConfig())
    assert "symbolic-link" in str(symlink_error.value)

    encrypted_path = _archive(tmp_path / "encrypted.docx", [("word/document.xml", "x")])
    payload = bytearray(encrypted_path.read_bytes())
    local = payload.index(b"PK\x03\x04")
    central = payload.index(b"PK\x01\x02")
    payload[local + 6] |= 0x01
    payload[central + 8] |= 0x01
    encrypted_path.write_bytes(payload)
    with pytest.raises(ParserError) as encrypted_error:
        validate_zip_archive(encrypted_path, ParserConfig())
    assert encrypted_error.value.code is ParserErrorCode.ENCRYPTED_CONTENT


def test_zip_preflight_accepts_small_safe_container(tmp_path):
    path = _archive(
        tmp_path / "safe.docx",
        [("[Content_Types].xml", "<Types/>") , ("word/document.xml", "<doc/>")],
    )
    diagnostics = validate_zip_archive(path, ParserConfig())
    assert diagnostics["member_count"] == 2
    assert diagnostics["uncompressed_bytes"] > 0
