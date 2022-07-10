
#include "ptrarr.h"

#ifdef CHECK

#include <check.h>

#define SUITE_NAME "ptrarr"

START_TEST(test_init)
{
    ptrarr_t arr;
    ptrarr_init(&arr);
    ck_assert(arr.len == 0);
    ptrarr_clear(&arr);
}
END_TEST

START_TEST(test_push)
{
    // make an array with test values
    int nums[20] = {20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1};
    ptrarr_t arr;
    ptrarr_init(&arr);
    ptrarr_push(&arr,nums+7);
    ck_assert(arr.len == 1);
    ck_assert(arr.array[0] == nums+7);
    ptrarr_push(&arr,nums+2);
    ck_assert(arr.len == 2);
    ck_assert(arr.array[0] == nums+7);
    ck_assert(arr.array[1] == nums+2);
    for (int i = 10; i < 20; ++i)
        ptrarr_push(&arr,nums+i);
    ck_assert(arr.len == 12);
    ck_assert(arr.array[0] == nums+7);
    ck_assert(arr.array[1] == nums+2);
    for (int i = 2; i < 12; ++i)
        ck_assert(arr.array[i] == nums+(i+8));
    ptrarr_clear(&arr);
}
END_TEST

START_TEST(test_clear)
{
    ptrarr_t arr;
    ptrarr_init(&arr);
    for (int i = 0; i < 10; ++i)
        ptrarr_push(&arr,NULL);
    ptrarr_clear(&arr);
    ck_assert(arr.len == 0);
    // clearing an already cleared array should be fine
    ptrarr_clear(&arr);
    ck_assert(arr.len == 0);
}
END_TEST

START_TEST(test_lengths)
{
    ptrarr_t arr;
    ptrarr_init(&arr);
    for (int i = 0; i < 10; ++i)
        ptrarr_push(&arr,NULL);
    ck_assert(arr.len == 10);
    ck_assert(arr.alloc >= 10);
    ptrarr_clear(&arr);
}
END_TEST

Suite *suite()
{
    Suite *s;
    TCase *tc;
    s = suite_create(SUITE_NAME);
    tc = tcase_create(SUITE_NAME);
    // TEST CASE LIST START
    tcase_add_test(tc,test_init);
    tcase_add_test(tc,test_push);
    tcase_add_test(tc,test_clear);
    tcase_add_test(tc,test_lengths);
    // TEST CASE LIST END
    suite_add_tcase(s,tc);
    return s;
}

int main(int argc, char **argv)
{
    int failed;
    Suite *s;
    SRunner *sr;
    s = suite();
    sr = srunner_create(s);
    srunner_run_all(sr,CK_VERBOSE);
    failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return failed == 0;
}

#endif
