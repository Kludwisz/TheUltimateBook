## What is this?

The item found using the CUDA bruteforce and discrete log steps will be very far away from the initial world seed we set. This becomes an issue as we can't directly check the exact fishing loot sequence leading up to that state.
Instead, we need some way to *prove* that *any* sequence of items starting at the particular world seed will end up generating the target book.
That's the job of this module.