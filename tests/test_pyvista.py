# -*- coding: utf-8 -*-


def test_arguments(testdir):
    """Test pytest arguments"""
    testdir.makepyfile("""
        def test_sth(verify_image_cache):
            assert verify_image_cache.reset_image_cache
            assert verify_image_cache.ignore_image_cache
            assert verify_image_cache.fail_extra_image_cache
           
    """)
    result = testdir.runpytest(
        "--reset_image_cache",
        "--ignore_image_cache",
        "--fail_extra_image_cache",
        
    )
    print(result.stdout)

