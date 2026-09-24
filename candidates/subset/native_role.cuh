#pragma once
// Role is identical in the nvcc host and device passes. Regeneration explicitly
// disables the native module; the fixed benchmark command selects the carrier.
#ifndef QSB_NATIVE_MODULE
#define QSB_NATIVE_MODULE 0
#endif
#ifndef QSB_HOST_CARRIER
#define QSB_HOST_CARRIER QSB_NATIVE_MODULE
#endif
#if (QSB_NATIVE_MODULE != 0 && QSB_NATIVE_MODULE != 1) || QSB_HOST_CARRIER != QSB_NATIVE_MODULE
#error "Select carrier/module together, or disable both for full-source regeneration"
#endif
#if QSB_HOST_CARRIER
#define QSB_ROLE_HD
#else
#define QSB_ROLE_HD __host__ __device__
#endif
