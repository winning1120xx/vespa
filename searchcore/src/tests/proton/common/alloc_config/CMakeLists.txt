# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
vespa_add_executable(searchcore_alloc_config_test_app TEST
    SOURCES
    alloc_config_test.cpp
    DEPENDS
    searchcore_pcommon
    GTest::gtest
)
vespa_add_test(NAME searchcore_alloc_config_test_app COMMAND searchcore_alloc_config_test_app)
