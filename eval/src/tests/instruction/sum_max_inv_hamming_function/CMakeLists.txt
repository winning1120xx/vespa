# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
vespa_add_executable(eval_sum_max_inv_hamming_function_test_app TEST
    SOURCES
    sum_max_inv_hamming_function_test.cpp
    DEPENDS
    vespaeval
    GTest::gtest
)
vespa_add_test(NAME eval_sum_max_inv_hamming_function_test_app COMMAND eval_sum_max_inv_hamming_function_test_app)
