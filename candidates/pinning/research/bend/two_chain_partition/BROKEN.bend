import Base
import ./LAWS.bend as Laws

# Deliberate fault: chunk 8 is dropped from the second chain. This file must
# fail to check; it is a sensitivity test, not a proof dependency.
def Laws.frontier_15_split(p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14):
  ?dropped_chunk_8
