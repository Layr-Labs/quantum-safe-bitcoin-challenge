import Base

# Packed product-tree locations used by ZLAB_TREE=3 for n=256.  During the
# downward pass, parent inverses live in the next packed level while every lane
# reads a sibling from the current level before overwriting its own child slot.

def level_offset(+count: U32) -> U32:
  (512 - (2 * count : U32) : U32)

def parent_slot(+count: U32, +tid: U32) -> U32:
  +half = (count / 2 : U32)
  (level_offset(count) + count + (tid % half : U32) : U32)

def sibling_choose(lower: Bool, +offset: U32, +tid: U32, +half: U32) -> U32:
  match lower:
    case True{}:
      (offset + tid + half : U32)
    case False{}:
      (offset + tid - half : U32)

def sibling_slot(+count: U32, +tid: U32) -> U32:
  +half = (count / 2 : U32)
  sibling_choose(U32.is_lt(tid, half), level_offset(count), tid, half)

def child_slot(+count: U32, +tid: U32) -> U32:
  (level_offset(count) + tid : U32)

law root_children:
  {(level_offset(2), (level_offset(2) + 1 : U32)) == (508, 509) : U32 & U32}

law cross_warp_boundary:
  {(parent_slot(128, 127), sibling_slot(128, 127), child_slot(128, 127)) == (447, 319, 383) : U32 & U32 & U32}

law warp_boundary:
  {(parent_slot(32, 31), sibling_slot(32, 31), child_slot(32, 31)) == (495, 463, 479) : U32 & U32 & U32}

law root_child_mutation_rejected:
  {(parent_slot(4, 3), sibling_slot(4, 3), child_slot(4, 3)) == (509, 505, 507) : U32 & U32 & U32}
