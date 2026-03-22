// License unlock for Spatial ACIS/Interop
#include "license.hxx"
#include "spa_unlock_result.hxx"
#include "spatial_license.h"

void unlock_license() {
    spa_unlock_result out = spa_unlock_products(SPATIAL_LICENSE);
    if (out.get_state() != SPA_UNLOCK_PASS && out.get_state() != SPA_UNLOCK_PASS_WARN) {
        fprintf(stderr, "ACIS license unlock failed\n");
    }
}
