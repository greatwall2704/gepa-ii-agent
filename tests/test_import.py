def test_package_import():
    """
    Ensures the 'gepa' package can be imported.
    """
    try:
        import gepa
    except ImportError as e:
        assert False, f"Failed to import the 'gepa' package: {e}"
