# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
vespa_add_executable(config_configholder_test_app TEST
    SOURCES
    configholder_test.cpp
    DEPENDS
    vespa_config
    GTest::gtest
)
vespa_add_test(NAME config_configholder_test_app COMMAND config_configholder_test_app)
