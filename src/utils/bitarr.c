
#include "bitarr.h"

#ifdef CHECK

#include <check.h>
#include <stdint.h>

#define SUITE_NAME "bitarr"

START_TEST(test_BIT_GET)
{
    uint8_t  a = 0x1C;
    uint16_t b = 0xBEEF;
    uint32_t c = 0xDEADC0DE;
    uint64_t d = 0x1337DEAD1337C0DE;
    ck_assert(BIT_GET(a,2));
    ck_assert(!BIT_GET(a,5));
    ck_assert(BIT_GET(b,9));
    ck_assert(!BIT_GET(b,14));
    ck_assert(BIT_GET(c,25));
    ck_assert(!BIT_GET(c,11));
    ck_assert(BIT_GET(d,42));
    ck_assert(!BIT_GET(d,59));
}
END_TEST

START_TEST(test_BIT_SET)
{
    uint8_t  a = 0x1C;
    uint16_t b = 0xBEEF;
    uint32_t c = 0xDEADC0DE;
    uint64_t d = 0x1337DEAD1337C0DE;
    BIT_SET(a,2);
    ck_assert_uint_eq(a,0x1C);
    BIT_SET(a,5);
    ck_assert_uint_eq(a,0x3C);
    BIT_SET(b,9);
    ck_assert_uint_eq(b,0xBEEF);
    BIT_SET(b,14);
    ck_assert_uint_eq(b,0xFEEF);
    BIT_SET(c,25);
    ck_assert_uint_eq(c,0xDEADC0DE);
    BIT_SET(c,11);
    ck_assert_uint_eq(c,0xDEADC8DE);
    BIT_SET(d,42);
    ck_assert_uint_eq(d,0x1337DEAD1337C0DE);
    BIT_SET(d,59);
    ck_assert_uint_eq(d,0x1B37DEAD1337C0DE);
}
END_TEST

START_TEST(test_BIT_UNSET)
{
    uint8_t  a = 0x1C;
    uint16_t b = 0xBEEF;
    uint32_t c = 0xDEADC0DE;
    uint64_t d = 0x1337DEAD1337C0DE;
    BIT_UNSET(a,2);
    ck_assert_uint_eq(a,0x18);
    BIT_UNSET(a,5);
    ck_assert_uint_eq(a,0x18);
    BIT_UNSET(b,9);
    ck_assert_uint_eq(b,0xBCEF);
    BIT_UNSET(b,14);
    ck_assert_uint_eq(b,0xBCEF);
    BIT_UNSET(c,25);
    ck_assert_uint_eq(c,0xDCADC0DE);
    BIT_UNSET(c,11);
    ck_assert_uint_eq(c,0xDCADC0DE);
    BIT_UNSET(d,42);
    ck_assert_uint_eq(d,0x1337DAAD1337C0DE);
    BIT_UNSET(d,59);
    ck_assert_uint_eq(d,0x1337DAAD1337C0DE);
}
END_TEST

Suite *sudoku_suite()
{
    Suite *s;
    TCase *tc;
    s = suite_create(SUITE_NAME);
    tc = tcase_create(SUITE_NAME);
    // TEST CASE LIST START
    tcase_add_test(tc,test_BIT_GET);
    tcase_add_test(tc,test_BIT_SET);
    tcase_add_test(tc,test_BIT_UNSET);
    // TEST CASE LIST END
    suite_add_tcase(s,tc);
    return s;
}

int main(int argc, char **argv)
{
    int failed;
    Suite *s;
    SRunner *sr;
    s = sudoku_suite();
    sr = srunner_create(s);
    srunner_run_all(sr,CK_VERBOSE);
    failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return failed == 0;
}

#endif
