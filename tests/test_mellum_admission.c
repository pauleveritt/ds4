#include "ds4.h"
#include <stdio.h>

int main(void) {
    int failures = ds4_test_mellum_decode_contract_admission();
    if (failures) {
        fprintf(stderr, "mellum admission: %d failure(s)\n", failures);
        return 1;
    }
    puts("mellum admission: ok");
    return 0;
}
