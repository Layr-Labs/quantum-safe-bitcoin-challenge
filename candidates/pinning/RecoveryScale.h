/*
* This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
* Copyright (c) 2019 Jean Luc PONS.
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, version 3.
*
* This program is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
* General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
*/


/* Same homogeneous mixed-add formula, exposing Zout/Zin=V^3. */
#ifndef QSB_RECOVERY_SCALE_H
#define QSB_RECOVERY_SCALE_H
__device__ void qsb_point_add_with_z_factor(uint64_t *p1x, uint64_t *p1y, uint64_t *p1z, uint64_t *p2x, uint64_t *p2y, uint64_t *z_factor)
{
  uint64_t u[4];
  uint64_t v[4];

  uint64_t us2[4];
  uint64_t vs2[4];
  uint64_t vs3[4];

  uint64_t a[4];

  uint64_t us2w[4];
  uint64_t vs2v2[4];
  uint64_t vs3u2[4];
  uint64_t _2vs2v2[4];

  _ModMult(u, p2y, p1z);
  _ModMult(v, p2x, p1z);

  _ModSub256(u, u, p1y);
  _ModSub256(v, v, p1x);

  _ModSqr(us2, u);
  _ModSqr(vs2, v);

  _ModMult(vs3, vs2, v);
  _ModMult(us2w, us2, p1z);
  _ModMult(vs2v2, vs2, p1x);

  _ModAdd256(_2vs2v2, vs2v2, vs2v2);

  _ModSub256(a, us2w, vs3);
  _ModSub256(a, _2vs2v2);

  _ModMult(p1x, v, a);
  _ModMult(vs3u2, vs3, p1y);

  _ModSub256(p1y, vs2v2, a);
  _ModMult(p1y, p1y, u);

  _ModSub256(p1y, vs3u2);
  _ModMult(p1z, vs3, p1z);
  memcpy(z_factor, vs3, 32);
}

#endif
