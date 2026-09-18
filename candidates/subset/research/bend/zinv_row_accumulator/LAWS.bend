import Base

def full_shift(x: U32) -> U32:
  (x >> 4n : U32)

def split_shift(+x: U32) -> U32:
  lo = (x .&. 255 : U32)
  hi = (x >> 8n : U32)
  ((hi << 4n : U32) + (lo >> 4n : U32) : U32)

def truncated_shift(x: U32) -> U32:
  lo = (x .&. 255 : U32)
  (lo >> 4n : U32)

law boundary_split_preserves:
  {split_shift(258) == full_shift(258) : U32}

law truncation_mutation_rejected:
  {U32.is_ne(truncated_shift(258), full_shift(258)) == True{} : Bool}
