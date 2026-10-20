from forcebench.paths import ARCHIVE_ROOT, CODE_ROOT, DATA_ROOT, SCRATCH_ROOT


def test_formal_scratch_archive_and_code_roots_are_separate():
    roots = {CODE_ROOT, DATA_ROOT, SCRATCH_ROOT, ARCHIVE_ROOT}
    assert len(roots) == 4
    assert not DATA_ROOT.is_relative_to(CODE_ROOT)
    assert not SCRATCH_ROOT.is_relative_to(CODE_ROOT)
    assert not ARCHIVE_ROOT.is_relative_to(CODE_ROOT)
