#pragma once

namespace secp256k1::fast {

// Run comprehensive self-tests on the library
// Returns true if all tests pass, false otherwise
// Set verbose=true to see detailed test output
bool Selftest(bool verbose = false);

} // namespace secp256k1::fast
