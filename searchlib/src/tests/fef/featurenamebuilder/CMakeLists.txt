# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
vespa_add_executable(searchlib_featurenamebuilder_test_app TEST
    SOURCES
    featurenamebuilder_test.cpp
    DEPENDS
    vespa_searchlib
    GTest::gtest
)
vespa_add_test(NAME searchlib_featurenamebuilder_test_app COMMAND searchlib_featurenamebuilder_test_app)
