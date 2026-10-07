

#pragma once
#if QSB_U32_TRIP == 3
__device__ __forceinline__ void qsb_pp_family(
    uint64_t *PPP, uint64_t *Q, uint64_t *ZZ,
    const uint64_t *U2, const uint64_t *X1) {
#ifdef __CUDA_ARCH__
    uint64_t p0,p1,p2,p3,q0,q1,q2,q3,z0,z1,z2,z3;
    asm(
"{\n"
"	.reg .u32 iu0,iu1,iu2,iu3,iu4,iu5,iu6,iu7;\n"
"	.reg .u32 ix0,ix1,ix2,ix3,ix4,ix5,ix6,ix7;\n"
"	.reg .u32 iz0,iz1,iz2,iz3,iz4,iz5,iz6,iz7;\n"
"	.reg .u32 p0,p1,p2,p3,p4,p5,p6,p7;\n"
"	.reg .u32 pp0,pp1,pp2,pp3,pp4,pp5,pp6,pp7;\n"
"	.reg .u32 op0,op1,op2,op3,op4,op5,op6,op7;\n"
"	.reg .u32 oq0,oq1,oq2,oq3,oq4,oq5,oq6,oq7;\n"
"	.reg .u32 oz0,oz1,oz2,oz3,oz4,oz5,oz6,oz7;\n"
"	.reg .u32 bm;\n"
"	mov.b64 {iu0,iu1}, %12;\n"
"	mov.b64 {iu2,iu3}, %13;\n"
"	mov.b64 {iu4,iu5}, %14;\n"
"	mov.b64 {iu6,iu7}, %15;\n"
"	mov.b64 {ix0,ix1}, %16;\n"
"	mov.b64 {ix2,ix3}, %17;\n"
"	mov.b64 {ix4,ix5}, %18;\n"
"	mov.b64 {ix6,ix7}, %19;\n"
"	mov.b64 {iz0,iz1}, %20;\n"
"	mov.b64 {iz2,iz3}, %21;\n"
"	mov.b64 {iz4,iz5}, %22;\n"
"	mov.b64 {iz6,iz7}, %23;\n"
"	sub.cc.u32 p0, iu0, ix0;\n"
"	subc.cc.u32 p1, iu1, ix1;\n"
"	subc.cc.u32 p2, iu2, ix2;\n"
"	subc.cc.u32 p3, iu3, ix3;\n"
"	subc.cc.u32 p4, iu4, ix4;\n"
"	subc.cc.u32 p5, iu5, ix5;\n"
"	subc.cc.u32 p6, iu6, ix6;\n"
"	subc.cc.u32 p7, iu7, ix7;\n"
"	subc.u32 bm, 0, 0;\n"
"	mad.lo.u32 p0, bm, 977, p0;\n"
"	add.u32 p1, p1, bm;\n"
"\n"
"	\n"
"	.reg .u64 Se2,Se4,Se6,Se8,Se10,Se12,So1,So3,So5,So7,So9,So11,So13,St;\n"
"	.reg .u32 Secy,Socy,Se14,So15;\n"
"	.reg .u32 Sx0,Sx1,Sx2,Sx3,Sx4,Sx5,Sx6,Sx7,Sx8,Sx9,Sx10,Sx11,Sx12,Sx13,Sx14,Sx15;\n"
"	.reg .u32 Sy2,Sy3,Sy4,Sy5,Sy6,Sy7,Sy8,Sy9,Sy10,Sy11,Sy12,Sy13,Sy14;\n"
"	.reg .u64 Sd0,Sd1,Sd2,Sd3,Sd4,Sd5,Sd6,Sd7;\n"
"	.reg .u64 Sodd_t;\n"
"mul.wide.u32 Se6, p2, p4;\n"
"mul.wide.u32 So7, p3, p4;\n"
"mul.wide.u32 Se8, p3, p5;\n"
"mul.wide.u32 So5, p2, p3;\n"
"mul.wide.u32 Se4, p1, p3;\n"
"mul.wide.u32 Sodd_t, p2, p5;\n"
"mul.wide.u32 St, p1, p5;\n"
"add.cc.u64 So7, So7, Sodd_t;\n"
"mul.wide.u32 Sodd_t, p4, p5;\n"
"addc.u64 So9, Sodd_t, 0;\n"
"add.cc.u64 Se6, Se6, St;\n"
"mul.wide.u32 St, p2, p6;\n"
"addc.cc.u64 Se8, Se8, St;\n"
"mul.wide.u32 St, p4, p6;\n"
"addc.u64 Se10, St, 0;\n"
"mul.wide.u32 So3, p1, p2;\n"
"mul.wide.u32 Se2, p0, p2;\n"
"mul.wide.u32 Sodd_t, p1, p4;\n"
"mul.wide.u32 St, p0, p4;\n"
"add.cc.u64 So5, So5, Sodd_t;\n"
"mul.wide.u32 Sodd_t, p1, p6;\n"
"addc.cc.u64 So7, So7, Sodd_t;\n"
"mul.wide.u32 Sodd_t, p3, p6;\n"
"addc.cc.u64 So9, So9, Sodd_t;\n"
"mul.wide.u32 Sodd_t, p5, p6;\n"
"addc.u64 So11, Sodd_t, 0;\n"
"add.cc.u64 Se4, Se4, St;\n"
"mul.wide.u32 St, p0, p6;\n"
"addc.cc.u64 Se6, Se6, St;\n"
"mul.wide.u32 St, p1, p7;\n"
"addc.cc.u64 Se8, Se8, St;\n"
"mul.wide.u32 St, p3, p7;\n"
"addc.cc.u64 Se10, Se10, St;\n"
"mul.wide.u32 St, p5, p7;\n"
"addc.u64 Se12, St, 0;\n"
"mul.wide.u32 So1, p0, p1;\n"
"mul.wide.u32 Sodd_t, p0, p3;\n"
"add.cc.u64 So3, So3, Sodd_t;\n"
"mul.wide.u32 Sodd_t, p0, p5;\n"
"addc.cc.u64 So5, So5, Sodd_t;\n"
"mul.wide.u32 Sodd_t, p0, p7;\n"
"addc.cc.u64 So7, So7, Sodd_t;\n"
"mul.wide.u32 Sodd_t, p2, p7;\n"
"addc.cc.u64 So9, So9, Sodd_t;\n"
"mul.wide.u32 Sodd_t, p4, p7;\n"
"addc.cc.u64 So11, So11, Sodd_t;\n"
"mul.wide.u32 Sodd_t, p6, p7;\n"
"addc.u64 So13, Sodd_t, 0;\n"
"mov.u32 Sx0, 0;\n"
"	mov.b64 {Sx2,Sx3}, Se2; mov.b64 {Sx4,Sx5}, Se4; mov.b64 {Sx6,Sx7}, Se6;\n"
"	mov.b64 {Sx8,Sx9}, Se8; mov.b64 {Sx10,Sx11}, Se10; mov.b64 {Sx12,Sx13}, Se12;\n"
"	mov.u32 Sx14, 0; mov.u32 Sx15, 0;\n"
"	mov.b64 {Sx1,Sy2}, So1; mov.b64 {Sy3,Sy4}, So3; mov.b64 {Sy5,Sy6}, So5;\n"
"	mov.b64 {Sy7,Sy8}, So7; mov.b64 {Sy9,Sy10}, So9; mov.b64 {Sy11,Sy12}, So11; mov.b64 {Sy13,Sy14}, So13;\n"
"	add.cc.u32 Sx2, Sx2, Sy2; addc.cc.u32 Sx3, Sx3, Sy3;\n"
"	addc.cc.u32 Sx4, Sx4, Sy4; addc.cc.u32 Sx5, Sx5, Sy5; addc.cc.u32 Sx6, Sx6, Sy6;\n"
"	addc.cc.u32 Sx7, Sx7, Sy7; addc.cc.u32 Sx8, Sx8, Sy8; addc.cc.u32 Sx9, Sx9, Sy9;\n"
"	addc.cc.u32 Sx10, Sx10, Sy10; addc.cc.u32 Sx11, Sx11, Sy11; addc.cc.u32 Sx12, Sx12, Sy12;\n"
"	addc.cc.u32 Sx13, Sx13, Sy13; addc.cc.u32 Sx14, Sx14, Sy14;\n"
"addc.u32 Sx15, Sx15, 0;\n"
"shf.l.wrap.b32 Sx15, Sx14, Sx15, 1; shf.l.wrap.b32 Sx14, Sx13, Sx14, 1;\n"
"	shf.l.wrap.b32 Sx13, Sx12, Sx13, 1; shf.l.wrap.b32 Sx12, Sx11, Sx12, 1;\n"
"	shf.l.wrap.b32 Sx11, Sx10, Sx11, 1; shf.l.wrap.b32 Sx10, Sx9, Sx10, 1;\n"
"	shf.l.wrap.b32 Sx9, Sx8, Sx9, 1; shf.l.wrap.b32 Sx8, Sx7, Sx8, 1;\n"
"	shf.l.wrap.b32 Sx7, Sx6, Sx7, 1; shf.l.wrap.b32 Sx6, Sx5, Sx6, 1;\n"
"	shf.l.wrap.b32 Sx5, Sx4, Sx5, 1; shf.l.wrap.b32 Sx4, Sx3, Sx4, 1;\n"
"	shf.l.wrap.b32 Sx3, Sx2, Sx3, 1; shf.l.wrap.b32 Sx2, Sx1, Sx2, 1;\n"
"	shl.b32 Sx1, Sx1, 1;\n"
"	 mov.b64 Sd1, {Sx2,Sx3}; mov.b64 Sd2, {Sx4,Sx5}; mov.b64 Sd3, {Sx6,Sx7};\n"
"	mov.b64 Sd4, {Sx8,Sx9}; mov.b64 Sd5, {Sx10,Sx11}; mov.b64 Sd6, {Sx12,Sx13}; mov.b64 Sd7, {Sx14,Sx15};\n"
"	mul.wide.u32 Sd0, p0, p0; { .reg .u32 Sq0l,Sq0h; mov.b64 {Sq0l,Sq0h}, Sd0; add.cc.u32 Sq0h, Sq0h, Sx1; mov.b64 Sd0, {Sq0l,Sq0h}; }\n"
"	mul.wide.u32 St, p1, p1; addc.cc.u64 Sd1, Sd1, St;\n"
"	mul.wide.u32 St, p2, p2; addc.cc.u64 Sd2, Sd2, St;\n"
"	mul.wide.u32 St, p3, p3; addc.cc.u64 Sd3, Sd3, St;\n"
"	mul.wide.u32 St, p4, p4; addc.cc.u64 Sd4, Sd4, St;\n"
"	mul.wide.u32 St, p5, p5; addc.cc.u64 Sd5, Sd5, St;\n"
"	mul.wide.u32 St, p6, p6; addc.cc.u64 Sd6, Sd6, St;\n"
"	mul.wide.u32 St, p7, p7; addc.u64 Sd7, Sd7, St;\n"
"		mov.b64 {Sx8,Sx9}, Sd4; mov.b64 {Sx10,Sx11}, Sd5; mov.b64 {Sx12,Sx13}, Sd6; mov.b64 {Sx14,Sx15}, Sd7;\n"
"	.reg .u64 Sf0,Sf1,Sf2,Sf3,Sg0,Sg1,Sg2,Sg3;\n"
"	.reg .u32 Sf8,Sg8,Sz8,Sz9,Sw0,Sw1,Sw2,Sw3,Sw4,Sw5,Sw6,Sw7,Sm0,Sm1,Sm2;\n"
"	mul.wide.u32 St, Sx8, 977;  add.cc.u64  Sf0, Sd0, St;\n"
"	mul.wide.u32 St, Sx10, 977; addc.cc.u64 Sf1, Sd1, St;\n"
"	mul.wide.u32 St, Sx12, 977; addc.cc.u64 Sf2, Sd2, St;\n"
"	mul.wide.u32 St, Sx14, 977; addc.cc.u64 Sf3, Sd3, St;\n"
"	addc.u32 Sf8, 0, 0;\n"
"	mul.wide.u32 St, Sx9, 977;  add.cc.u64  Sg0, Sd4, St;\n"
"	mul.wide.u32 St, Sx11, 977; addc.cc.u64 Sg1, Sd5, St;\n"
"	mul.wide.u32 St, Sx13, 977; addc.cc.u64 Sg2, Sd6, St;\n"
"	mul.wide.u32 St, Sx15, 977; addc.cc.u64 Sg3, Sd7, St;\n"
"	mov.b64 {pp0,pp1}, Sf0; mov.b64 {pp2,pp3}, Sf1; mov.b64 {pp4,pp5}, Sf2; mov.b64 {pp6,pp7}, Sf3;\n"
"	mov.b64 {Sw0,Sw1}, Sg0; mov.b64 {Sw2,Sw3}, Sg1; mov.b64 {Sw4,Sw5}, Sg2; mov.b64 {Sw6,Sw7}, Sg3;\n"
"	add.cc.u32 pp1, pp1, Sw0; addc.cc.u32 pp2, pp2, Sw1; addc.cc.u32 pp3, pp3, Sw2;\n"
"	addc.cc.u32 pp4, pp4, Sw3; addc.cc.u32 pp5, pp5, Sw4; addc.cc.u32 pp6, pp6, Sw5;\n"
"	addc.cc.u32 pp7, pp7, Sw6; addc.cc.u32 Sz8, Sf8, Sw7;\n"
"	{ .reg .u64 Ssfa,Ssft; .reg .u32 Ssfl,Ssfh;\n"
"\n"
"\n"
"mul.wide.u32 Ssft, Sz8, 977;\n"
"mov.b64 Ssfa, {pp0, pp1}; add.cc.u64 Ssfa, Ssfa, Ssft; addc.u32 pp2, pp2, 0; mov.b64 {Ssfl, Ssfh}, Ssfa;\n"
"\n"
"\n"
"mov.u32 pp0, Ssfl;\n"
"add.cc.u32 pp1, Ssfh, Sz8; addc.u32 pp2, pp2, 0;\n"
" }\n"
"\n"
"\n"
"\n"
"	\n"
"	.reg .u64 Ae0,Ae1,Ae2,Ae3,Ae4,Ae5,Ae6,Ae7,Ao0,Ao1,Ao2,Ao3,Ao4,Ao5,Ao6,At,Alc;\n"
"	.reg .u32 Acy,Ao15;\n"
"	.reg .u32 Ax0,Ax1,Ax2,Ax3,Ax4,Ax5,Ax6,Ax7,Ax8,Ax9,Ax10,Ax11,Ax12,Ax13,Ax14,Ax15;\n"
"	.reg .u32 Ay1,Ay2,Ay3,Ay4,Ay5,Ay6,Ay7,Ay8,Ay9,Ay10,Ay11,Ay12,Ay13,Ay14;\n"
"	.reg .u64 Aodd_t,Aodd_lc; .reg .u32 Aodd_cy;\n"
"mul.wide.u32 Ae0, pp0, p0;\n"
"mul.wide.u32 Ao0, pp0, p1;\n"
"mul.wide.u32 Ae1, pp0, p2;\n"
"mul.wide.u32 Ao1, pp0, p3;\n"
"mul.wide.u32 Ae2, pp0, p4;\n"
"mul.wide.u32 Ao2, pp0, p5;\n"
"mul.wide.u32 Ae3, pp0, p6;\n"
"mul.wide.u32 Ao3, pp0, p7;\n"
"mul.wide.u32 At, pp1, p1;\n"
"mul.wide.u32 Aodd_t, pp1, p0;\n"
"add.cc.u64 Ae1, Ae1, At;\n"
"mul.wide.u32 At, pp1, p3;\n"
"addc.cc.u64 Ae2, Ae2, At;\n"
"mul.wide.u32 At, pp1, p5;\n"
"addc.cc.u64 Ae3, Ae3, At;\n"
"mul.wide.u32 At, pp1, p7;\n"
"addc.u64 Ae4, At, 0;\n"
"add.cc.u64 Ao0, Ao0, Aodd_t;\n"
"mul.wide.u32 Aodd_t, pp1, p2;\n"
"addc.cc.u64 Ao1, Ao1, Aodd_t;\n"
"mul.wide.u32 Aodd_t, pp1, p4;\n"
"addc.cc.u64 Ao2, Ao2, Aodd_t;\n"
"mul.wide.u32 Aodd_t, pp1, p6;\n"
"addc.cc.u64 Ao3, Ao3, Aodd_t;\n"
"addc.u32 Aodd_cy, 0, 0;\n"
"mul.wide.u32 At, pp2, p0;\n"
"add.cc.u64 Ae1, Ae1, At;\n"
"mul.wide.u32 At, pp2, p2;\n"
"addc.cc.u64 Ae2, Ae2, At;\n"
"mul.wide.u32 At, pp2, p4;\n"
"addc.cc.u64 Ae3, Ae3, At;\n"
"mul.wide.u32 At, pp2, p6;\n"
"addc.cc.u64 Ae4, Ae4, At;\n"
"addc.u32 Acy, 0, 0;\n"
"mul.wide.u32 Aodd_t, pp2, p1;\n"
"mov.b64 Aodd_lc, {Aodd_cy, Acy};\n"
"add.cc.u64 Ao1, Ao1, Aodd_t;\n"
"mul.wide.u32 Aodd_t, pp2, p3;\n"
"addc.cc.u64 Ao2, Ao2, Aodd_t;\n"
"mul.wide.u32 Aodd_t, pp2, p5;\n"
"addc.cc.u64 Ao3, Ao3, Aodd_t;\n"
"mul.wide.u32 Aodd_t, pp2, p7;\n"
"addc.u64 Ao4, Aodd_t, Aodd_lc;\n"
"mul.wide.u32 At, pp3, p1;\n"
"mul.wide.u32 Aodd_t, pp3, p0;\n"
"add.cc.u64 Ae2, Ae2, At;\n"
"mul.wide.u32 At, pp3, p3;\n"
"addc.cc.u64 Ae3, Ae3, At;\n"
"mul.wide.u32 At, pp3, p5;\n"
"addc.cc.u64 Ae4, Ae4, At;\n"
"mul.wide.u32 At, pp3, p7;\n"
"addc.u64 Ae5, At, 0;\n"
"add.cc.u64 Ao1, Ao1, Aodd_t;\n"
"mul.wide.u32 Aodd_t, pp3, p2;\n"
"addc.cc.u64 Ao2, Ao2, Aodd_t;\n"
"mul.wide.u32 Aodd_t, pp3, p4;\n"
"addc.cc.u64 Ao3, Ao3, Aodd_t;\n"
"mul.wide.u32 Aodd_t, pp3, p6;\n"
"addc.cc.u64 Ao4, Ao4, Aodd_t;\n"
"addc.u32 Aodd_cy, 0, 0;\n"
"mul.wide.u32 At, pp4, p0;\n"
"add.cc.u64 Ae2, Ae2, At;\n"
"mul.wide.u32 At, pp4, p2;\n"
"addc.cc.u64 Ae3, Ae3, At;\n"
"mul.wide.u32 At, pp4, p4;\n"
"addc.cc.u64 Ae4, Ae4, At;\n"
"mul.wide.u32 At, pp4, p6;\n"
"addc.cc.u64 Ae5, Ae5, At;\n"
"addc.u32 Acy, 0, 0;\n"
"mul.wide.u32 Aodd_t, pp4, p1;\n"
"mov.b64 Aodd_lc, {Aodd_cy, Acy};\n"
"add.cc.u64 Ao2, Ao2, Aodd_t;\n"
"mul.wide.u32 Aodd_t, pp4, p3;\n"
"addc.cc.u64 Ao3, Ao3, Aodd_t;\n"
"mul.wide.u32 Aodd_t, pp4, p5;\n"
"addc.cc.u64 Ao4, Ao4, Aodd_t;\n"
"mul.wide.u32 Aodd_t, pp4, p7;\n"
"addc.u64 Ao5, Aodd_t, Aodd_lc;\n"
"mul.wide.u32 At, pp5, p1;\n"
"mul.wide.u32 Aodd_t, pp5, p0;\n"
"add.cc.u64 Ae3, Ae3, At;\n"
"mul.wide.u32 At, pp5, p3;\n"
"addc.cc.u64 Ae4, Ae4, At;\n"
"mul.wide.u32 At, pp5, p5;\n"
"addc.cc.u64 Ae5, Ae5, At;\n"
"mul.wide.u32 At, pp5, p7;\n"
"addc.u64 Ae6, At, 0;\n"
"add.cc.u64 Ao2, Ao2, Aodd_t;\n"
"mul.wide.u32 Aodd_t, pp5, p2;\n"
"addc.cc.u64 Ao3, Ao3, Aodd_t;\n"
"mul.wide.u32 Aodd_t, pp5, p4;\n"
"addc.cc.u64 Ao4, Ao4, Aodd_t;\n"
"mul.wide.u32 Aodd_t, pp5, p6;\n"
"addc.cc.u64 Ao5, Ao5, Aodd_t;\n"
"addc.u32 Aodd_cy, 0, 0;\n"
"mul.wide.u32 At, pp6, p0;\n"
"add.cc.u64 Ae3, Ae3, At;\n"
"mul.wide.u32 At, pp6, p2;\n"
"addc.cc.u64 Ae4, Ae4, At;\n"
"mul.wide.u32 At, pp6, p4;\n"
"addc.cc.u64 Ae5, Ae5, At;\n"
"mul.wide.u32 At, pp6, p6;\n"
"addc.cc.u64 Ae6, Ae6, At;\n"
"addc.u32 Acy, 0, 0;\n"
"mul.wide.u32 Aodd_t, pp6, p1;\n"
"mov.b64 Aodd_lc, {Aodd_cy, Acy};\n"
"add.cc.u64 Ao3, Ao3, Aodd_t;\n"
"mul.wide.u32 Aodd_t, pp6, p3;\n"
"addc.cc.u64 Ao4, Ao4, Aodd_t;\n"
"mul.wide.u32 Aodd_t, pp6, p5;\n"
"addc.cc.u64 Ao5, Ao5, Aodd_t;\n"
"mul.wide.u32 Aodd_t, pp6, p7;\n"
"addc.u64 Ao6, Aodd_t, Aodd_lc;\n"
"mul.wide.u32 At, pp7, p1;\n"
"mul.wide.u32 Aodd_t, pp7, p0;\n"
"add.cc.u64 Ae4, Ae4, At;\n"
"mul.wide.u32 At, pp7, p3;\n"
"addc.cc.u64 Ae5, Ae5, At;\n"
"mul.wide.u32 At, pp7, p5;\n"
"addc.cc.u64 Ae6, Ae6, At;\n"
"mul.wide.u32 At, pp7, p7;\n"
"addc.u64 Ae7, At, 0;\n"
"add.cc.u64 Ao3, Ao3, Aodd_t;\n"
"mul.wide.u32 Aodd_t, pp7, p2;\n"
"addc.cc.u64 Ao4, Ao4, Aodd_t;\n"
"mul.wide.u32 Aodd_t, pp7, p4;\n"
"addc.cc.u64 Ao5, Ao5, Aodd_t;\n"
"mul.wide.u32 Aodd_t, pp7, p6;\n"
"addc.cc.u64 Ao6, Ao6, Aodd_t;\n"
"mov.b64 {Ax14,Ax15}, Ae7; addc.u32 Ax15, Ax15, 0;\n"
"mov.b64 {Ax0,Ax1}, Ae0;\n"
"	mov.b64 {Ax2,Ax3}, Ae1;\n"
"	mov.b64 {Ax4,Ax5}, Ae2;\n"
"	mov.b64 {Ax6,Ax7}, Ae3;\n"
"	mov.b64 {Ax8,Ax9}, Ae4;\n"
"	mov.b64 {Ax10,Ax11}, Ae5;\n"
"	mov.b64 {Ax12,Ax13}, Ae6;\n"
"	\n"
"	mov.b64 {Ay1,Ay2}, Ao0;\n"
"	mov.b64 {Ay3,Ay4}, Ao1;\n"
"	mov.b64 {Ay5,Ay6}, Ao2;\n"
"	mov.b64 {Ay7,Ay8}, Ao3;\n"
"	mov.b64 {Ay9,Ay10}, Ao4;\n"
"	mov.b64 {Ay11,Ay12}, Ao5;\n"
"	mov.b64 {Ay13,Ay14}, Ao6;\n"
"	add.cc.u32 Ax1, Ax1, Ay1;\n"
"	addc.cc.u32 Ax2, Ax2, Ay2;\n"
"	addc.cc.u32 Ax3, Ax3, Ay3;\n"
"	addc.cc.u32 Ax4, Ax4, Ay4;\n"
"	addc.cc.u32 Ax5, Ax5, Ay5;\n"
"	addc.cc.u32 Ax6, Ax6, Ay6;\n"
"	addc.cc.u32 Ax7, Ax7, Ay7;\n"
"	addc.cc.u32 Ax8, Ax8, Ay8;\n"
"	addc.cc.u32 Ax9, Ax9, Ay9;\n"
"	addc.cc.u32 Ax10, Ax10, Ay10;\n"
"	addc.cc.u32 Ax11, Ax11, Ay11;\n"
"	addc.cc.u32 Ax12, Ax12, Ay12;\n"
"	addc.cc.u32 Ax13, Ax13, Ay13;\n"
"	addc.cc.u32 Ax14, Ax14, Ay14;\n"
"	addc.u32 Ax15, Ax15, 0;\n"
"	.reg .u64 Ar0,Ar1,Ar2,Ar3,Ah0,Ah1,Ah2,Ah3,Af0,Af1,Af2,Af3,Ag0,Ag1,Ag2,Ag3;\n"
"	.reg .u32 Af8,Ag8,Az8,Az9,Aw0,Aw1,Aw2,Aw3,Aw4,Aw5,Aw6,Aw7,Am0,Am1,Am2;\n"
"	mov.b64 Ar0, {Ax0,Ax1}; mov.b64 Ar1, {Ax2,Ax3}; mov.b64 Ar2, {Ax4,Ax5}; mov.b64 Ar3, {Ax6,Ax7};\n"
"	mov.b64 Ah0, {Ax8,Ax9}; mov.b64 Ah1, {Ax10,Ax11}; mov.b64 Ah2, {Ax12,Ax13}; mov.b64 Ah3, {Ax14,Ax15};\n"
"	mul.wide.u32 At, Ax8, 977;  add.cc.u64  Af0, Ar0, At;\n"
"	mul.wide.u32 At, Ax10, 977; addc.cc.u64 Af1, Ar1, At;\n"
"	mul.wide.u32 At, Ax12, 977; addc.cc.u64 Af2, Ar2, At;\n"
"	mul.wide.u32 At, Ax14, 977; addc.cc.u64 Af3, Ar3, At;\n"
"	mul.wide.u32 At, Ax9, 977;  add.cc.u64  Ag0, Ah0, At;\n"
"	mul.wide.u32 At, Ax11, 977; addc.cc.u64 Ag1, Ah1, At;\n"
"	mul.wide.u32 At, Ax13, 977; addc.cc.u64 Ag2, Ah2, At;\n"
"	mul.wide.u32 At, Ax15, 977; addc.cc.u64 Ag3, Ah3, At;\n"
"	/*rp*/\n"
"	mov.b64 {op0,op1}, Af0;\n"
"	mov.b64 {op2,op3}, Af1;\n"
"	mov.b64 {op4,op5}, Af2;\n"
"	mov.b64 {op6,op7}, Af3;\n"
"	mov.b64 {Aw0,Aw1}, Ag0;\n"
"	mov.b64 {Aw2,Aw3}, Ag1;\n"
"	mov.b64 {Aw4,Aw5}, Ag2;\n"
"	mov.b64 {Aw6,Aw7}, Ag3;\n"
"	add.cc.u32  op1, op1, Aw0;\n"
"	addc.cc.u32 op2, op2, Aw1;\n"
"	addc.cc.u32 op3, op3, Aw2;\n"
"	addc.cc.u32 op4, op4, Aw3;\n"
"	addc.cc.u32 op5, op5, Aw4;\n"
"	addc.cc.u32 op6, op6, Aw5;\n"
"	addc.cc.u32 op7, op7, Aw6;\n"
"	addc.u32 Az8, 0, Aw7;\n"
"	{ .reg .u64 Asfa,Asft; .reg .u32 Asfl,Asfh;\n"
"mul.wide.u32 Asft, Az8, 977;\n"
"mov.b64 Asfa, {op0, op1}; add.cc.u64 Asfa, Asfa, Asft; addc.u32 op2, op2, 0; mov.b64 {Asfl, Asfh}, Asfa;\n"
"\n"
"\n"
"mov.u32 op0, Asfl;\n"
"add.cc.u32 op1, Asfh, Az8; addc.u32 op2, op2, 0;\n"
" }\n"
"\n"
"\n"
"\n"
"	\n"
"	.reg .u64 Be0,Be1,Be2,Be3,Be4,Be5,Be6,Be7,Bo0,Bo1,Bo2,Bo3,Bo4,Bo5,Bo6,Bt,Blc;\n"
"	.reg .u32 Bcy,Bo15;\n"
"	.reg .u32 Bx0,Bx1,Bx2,Bx3,Bx4,Bx5,Bx6,Bx7,Bx8,Bx9,Bx10,Bx11,Bx12,Bx13,Bx14,Bx15;\n"
"	.reg .u32 By1,By2,By3,By4,By5,By6,By7,By8,By9,By10,By11,By12,By13,By14;\n"
"	.reg .u64 Bodd_t,Bodd_lc; .reg .u32 Bodd_cy;\n"
"mul.wide.u32 Be0, iu0, pp0;\n"
"mul.wide.u32 Bo0, iu0, pp1;\n"
"mul.wide.u32 Be1, iu0, pp2;\n"
"mul.wide.u32 Bo1, iu0, pp3;\n"
"mul.wide.u32 Be2, iu0, pp4;\n"
"mul.wide.u32 Bo2, iu0, pp5;\n"
"mul.wide.u32 Be3, iu0, pp6;\n"
"mul.wide.u32 Bo3, iu0, pp7;\n"
"mul.wide.u32 Bt, iu1, pp1;\n"
"mul.wide.u32 Bodd_t, iu1, pp0;\n"
"add.cc.u64 Be1, Be1, Bt;\n"
"mul.wide.u32 Bt, iu1, pp3;\n"
"addc.cc.u64 Be2, Be2, Bt;\n"
"mul.wide.u32 Bt, iu1, pp5;\n"
"addc.cc.u64 Be3, Be3, Bt;\n"
"mul.wide.u32 Bt, iu1, pp7;\n"
"addc.u64 Be4, Bt, 0;\n"
"add.cc.u64 Bo0, Bo0, Bodd_t;\n"
"mul.wide.u32 Bodd_t, iu1, pp2;\n"
"addc.cc.u64 Bo1, Bo1, Bodd_t;\n"
"mul.wide.u32 Bodd_t, iu1, pp4;\n"
"addc.cc.u64 Bo2, Bo2, Bodd_t;\n"
"mul.wide.u32 Bodd_t, iu1, pp6;\n"
"addc.cc.u64 Bo3, Bo3, Bodd_t;\n"
"addc.u32 Bodd_cy, 0, 0;\n"
"mul.wide.u32 Bt, iu2, pp0;\n"
"add.cc.u64 Be1, Be1, Bt;\n"
"mul.wide.u32 Bt, iu2, pp2;\n"
"addc.cc.u64 Be2, Be2, Bt;\n"
"mul.wide.u32 Bt, iu2, pp4;\n"
"addc.cc.u64 Be3, Be3, Bt;\n"
"mul.wide.u32 Bt, iu2, pp6;\n"
"addc.cc.u64 Be4, Be4, Bt;\n"
"addc.u32 Bcy, 0, 0;\n"
"mul.wide.u32 Bodd_t, iu2, pp1;\n"
"mov.b64 Bodd_lc, {Bodd_cy, Bcy};\n"
"add.cc.u64 Bo1, Bo1, Bodd_t;\n"
"mul.wide.u32 Bodd_t, iu2, pp3;\n"
"addc.cc.u64 Bo2, Bo2, Bodd_t;\n"
"mul.wide.u32 Bodd_t, iu2, pp5;\n"
"addc.cc.u64 Bo3, Bo3, Bodd_t;\n"
"mul.wide.u32 Bodd_t, iu2, pp7;\n"
"addc.u64 Bo4, Bodd_t, Bodd_lc;\n"
"mul.wide.u32 Bt, iu3, pp1;\n"
"mul.wide.u32 Bodd_t, iu3, pp0;\n"
"add.cc.u64 Be2, Be2, Bt;\n"
"mul.wide.u32 Bt, iu3, pp3;\n"
"addc.cc.u64 Be3, Be3, Bt;\n"
"mul.wide.u32 Bt, iu3, pp5;\n"
"addc.cc.u64 Be4, Be4, Bt;\n"
"mul.wide.u32 Bt, iu3, pp7;\n"
"addc.u64 Be5, Bt, 0;\n"
"add.cc.u64 Bo1, Bo1, Bodd_t;\n"
"mul.wide.u32 Bodd_t, iu3, pp2;\n"
"addc.cc.u64 Bo2, Bo2, Bodd_t;\n"
"mul.wide.u32 Bodd_t, iu3, pp4;\n"
"addc.cc.u64 Bo3, Bo3, Bodd_t;\n"
"mul.wide.u32 Bodd_t, iu3, pp6;\n"
"addc.cc.u64 Bo4, Bo4, Bodd_t;\n"
"addc.u32 Bodd_cy, 0, 0;\n"
"mul.wide.u32 Bt, iu4, pp0;\n"
"add.cc.u64 Be2, Be2, Bt;\n"
"mul.wide.u32 Bt, iu4, pp2;\n"
"addc.cc.u64 Be3, Be3, Bt;\n"
"mul.wide.u32 Bt, iu4, pp4;\n"
"addc.cc.u64 Be4, Be4, Bt;\n"
"mul.wide.u32 Bt, iu4, pp6;\n"
"addc.cc.u64 Be5, Be5, Bt;\n"
"addc.u32 Bcy, 0, 0;\n"
"mul.wide.u32 Bodd_t, iu4, pp1;\n"
"mov.b64 Bodd_lc, {Bodd_cy, Bcy};\n"
"add.cc.u64 Bo2, Bo2, Bodd_t;\n"
"mul.wide.u32 Bodd_t, iu4, pp3;\n"
"addc.cc.u64 Bo3, Bo3, Bodd_t;\n"
"mul.wide.u32 Bodd_t, iu4, pp5;\n"
"addc.cc.u64 Bo4, Bo4, Bodd_t;\n"
"mul.wide.u32 Bodd_t, iu4, pp7;\n"
"addc.u64 Bo5, Bodd_t, Bodd_lc;\n"
"mul.wide.u32 Bt, iu5, pp1;\n"
"mul.wide.u32 Bodd_t, iu5, pp0;\n"
"add.cc.u64 Be3, Be3, Bt;\n"
"mul.wide.u32 Bt, iu5, pp3;\n"
"addc.cc.u64 Be4, Be4, Bt;\n"
"mul.wide.u32 Bt, iu5, pp5;\n"
"addc.cc.u64 Be5, Be5, Bt;\n"
"mul.wide.u32 Bt, iu5, pp7;\n"
"addc.u64 Be6, Bt, 0;\n"
"add.cc.u64 Bo2, Bo2, Bodd_t;\n"
"mul.wide.u32 Bodd_t, iu5, pp2;\n"
"addc.cc.u64 Bo3, Bo3, Bodd_t;\n"
"mul.wide.u32 Bodd_t, iu5, pp4;\n"
"addc.cc.u64 Bo4, Bo4, Bodd_t;\n"
"mul.wide.u32 Bodd_t, iu5, pp6;\n"
"addc.cc.u64 Bo5, Bo5, Bodd_t;\n"
"addc.u32 Bodd_cy, 0, 0;\n"
"mul.wide.u32 Bt, iu6, pp0;\n"
"add.cc.u64 Be3, Be3, Bt;\n"
"mul.wide.u32 Bt, iu6, pp2;\n"
"addc.cc.u64 Be4, Be4, Bt;\n"
"mul.wide.u32 Bt, iu6, pp4;\n"
"addc.cc.u64 Be5, Be5, Bt;\n"
"mul.wide.u32 Bt, iu6, pp6;\n"
"addc.cc.u64 Be6, Be6, Bt;\n"
"addc.u32 Bcy, 0, 0;\n"
"mul.wide.u32 Bodd_t, iu6, pp1;\n"
"mov.b64 Bodd_lc, {Bodd_cy, Bcy};\n"
"add.cc.u64 Bo3, Bo3, Bodd_t;\n"
"mul.wide.u32 Bodd_t, iu6, pp3;\n"
"addc.cc.u64 Bo4, Bo4, Bodd_t;\n"
"mul.wide.u32 Bodd_t, iu6, pp5;\n"
"addc.cc.u64 Bo5, Bo5, Bodd_t;\n"
"mul.wide.u32 Bodd_t, iu6, pp7;\n"
"addc.u64 Bo6, Bodd_t, Bodd_lc;\n"
"mul.wide.u32 Bt, iu7, pp1;\n"
"mul.wide.u32 Bodd_t, iu7, pp0;\n"
"add.cc.u64 Be4, Be4, Bt;\n"
"mul.wide.u32 Bt, iu7, pp3;\n"
"addc.cc.u64 Be5, Be5, Bt;\n"
"mul.wide.u32 Bt, iu7, pp5;\n"
"addc.cc.u64 Be6, Be6, Bt;\n"
"mul.wide.u32 Bt, iu7, pp7;\n"
"addc.u64 Be7, Bt, 0;\n"
"add.cc.u64 Bo3, Bo3, Bodd_t;\n"
"mul.wide.u32 Bodd_t, iu7, pp2;\n"
"addc.cc.u64 Bo4, Bo4, Bodd_t;\n"
"mul.wide.u32 Bodd_t, iu7, pp4;\n"
"addc.cc.u64 Bo5, Bo5, Bodd_t;\n"
"mul.wide.u32 Bodd_t, iu7, pp6;\n"
"addc.cc.u64 Bo6, Bo6, Bodd_t;\n"
"mov.b64 {Bx14,Bx15}, Be7; addc.u32 Bx15, Bx15, 0;\n"
"mov.b64 {Bx0,Bx1}, Be0;\n"
"	mov.b64 {Bx2,Bx3}, Be1;\n"
"	mov.b64 {Bx4,Bx5}, Be2;\n"
"	mov.b64 {Bx6,Bx7}, Be3;\n"
"	mov.b64 {Bx8,Bx9}, Be4;\n"
"	mov.b64 {Bx10,Bx11}, Be5;\n"
"	mov.b64 {Bx12,Bx13}, Be6;\n"
"	\n"
"	mov.b64 {By1,By2}, Bo0;\n"
"	mov.b64 {By3,By4}, Bo1;\n"
"	mov.b64 {By5,By6}, Bo2;\n"
"	mov.b64 {By7,By8}, Bo3;\n"
"	mov.b64 {By9,By10}, Bo4;\n"
"	mov.b64 {By11,By12}, Bo5;\n"
"	mov.b64 {By13,By14}, Bo6;\n"
"	add.cc.u32 Bx1, Bx1, By1;\n"
"	addc.cc.u32 Bx2, Bx2, By2;\n"
"	addc.cc.u32 Bx3, Bx3, By3;\n"
"	addc.cc.u32 Bx4, Bx4, By4;\n"
"	addc.cc.u32 Bx5, Bx5, By5;\n"
"	addc.cc.u32 Bx6, Bx6, By6;\n"
"	addc.cc.u32 Bx7, Bx7, By7;\n"
"	addc.cc.u32 Bx8, Bx8, By8;\n"
"	addc.cc.u32 Bx9, Bx9, By9;\n"
"	addc.cc.u32 Bx10, Bx10, By10;\n"
"	addc.cc.u32 Bx11, Bx11, By11;\n"
"	addc.cc.u32 Bx12, Bx12, By12;\n"
"	addc.cc.u32 Bx13, Bx13, By13;\n"
"	addc.cc.u32 Bx14, Bx14, By14;\n"
"	addc.u32 Bx15, Bx15, 0;\n"
"	.reg .u64 Br0,Br1,Br2,Br3,Bh0,Bh1,Bh2,Bh3,Bf0,Bf1,Bf2,Bf3,Bg0,Bg1,Bg2,Bg3;\n"
"	.reg .u32 Bf8,Bg8,Bz8,Bz9,Bw0,Bw1,Bw2,Bw3,Bw4,Bw5,Bw6,Bw7,Bm0,Bm1,Bm2;\n"
"	mov.b64 Br0, {Bx0,Bx1}; mov.b64 Br1, {Bx2,Bx3}; mov.b64 Br2, {Bx4,Bx5}; mov.b64 Br3, {Bx6,Bx7};\n"
"	mov.b64 Bh0, {Bx8,Bx9}; mov.b64 Bh1, {Bx10,Bx11}; mov.b64 Bh2, {Bx12,Bx13}; mov.b64 Bh3, {Bx14,Bx15};\n"
"	mul.wide.u32 Bt, Bx8, 977;  add.cc.u64  Bf0, Br0, Bt;\n"
"	mul.wide.u32 Bt, Bx10, 977; addc.cc.u64 Bf1, Br1, Bt;\n"
"	mul.wide.u32 Bt, Bx12, 977; addc.cc.u64 Bf2, Br2, Bt;\n"
"	mul.wide.u32 Bt, Bx14, 977; addc.cc.u64 Bf3, Br3, Bt;\n"
"	mul.wide.u32 Bt, Bx9, 977;  add.cc.u64  Bg0, Bh0, Bt;\n"
"	mul.wide.u32 Bt, Bx11, 977; addc.cc.u64 Bg1, Bh1, Bt;\n"
"	mul.wide.u32 Bt, Bx13, 977; addc.cc.u64 Bg2, Bh2, Bt;\n"
"	mul.wide.u32 Bt, Bx15, 977; addc.cc.u64 Bg3, Bh3, Bt;\n"
"	/*rp*/\n"
"	mov.b64 {oq0,oq1}, Bf0;\n"
"	mov.b64 {oq2,oq3}, Bf1;\n"
"	mov.b64 {oq4,oq5}, Bf2;\n"
"	mov.b64 {oq6,oq7}, Bf3;\n"
"	mov.b64 {Bw0,Bw1}, Bg0;\n"
"	mov.b64 {Bw2,Bw3}, Bg1;\n"
"	mov.b64 {Bw4,Bw5}, Bg2;\n"
"	mov.b64 {Bw6,Bw7}, Bg3;\n"
"	add.cc.u32  oq1, oq1, Bw0;\n"
"	addc.cc.u32 oq2, oq2, Bw1;\n"
"	addc.cc.u32 oq3, oq3, Bw2;\n"
"	addc.cc.u32 oq4, oq4, Bw3;\n"
"	addc.cc.u32 oq5, oq5, Bw4;\n"
"	addc.cc.u32 oq6, oq6, Bw5;\n"
"	addc.cc.u32 oq7, oq7, Bw6;\n"
"	addc.u32 Bz8, 0, Bw7;\n"
"	{ .reg .u64 Bsfa,Bsft; .reg .u32 Bsfl,Bsfh;\n"
"mul.wide.u32 Bsft, Bz8, 977;\n"
"mov.b64 Bsfa, {oq0, oq1}; add.cc.u64 Bsfa, Bsfa, Bsft; addc.u32 oq2, oq2, 0; mov.b64 {Bsfl, Bsfh}, Bsfa;\n"
"\n"
"\n"
"mov.u32 oq0, Bsfl;\n"
"add.cc.u32 oq1, Bsfh, Bz8; addc.u32 oq2, oq2, 0;\n"
" }\n"
"\n"
"\n"
"\n"
"	\n"
"	.reg .u64 Ce0,Ce1,Ce2,Ce3,Ce4,Ce5,Ce6,Ce7,Co0,Co1,Co2,Co3,Co4,Co5,Co6,Ct,Clc;\n"
"	.reg .u32 Ccy,Co15;\n"
"	.reg .u32 Cx0,Cx1,Cx2,Cx3,Cx4,Cx5,Cx6,Cx7,Cx8,Cx9,Cx10,Cx11,Cx12,Cx13,Cx14,Cx15;\n"
"	.reg .u32 Cy1,Cy2,Cy3,Cy4,Cy5,Cy6,Cy7,Cy8,Cy9,Cy10,Cy11,Cy12,Cy13,Cy14;\n"
"	.reg .u64 Codd_t,Codd_lc; .reg .u32 Codd_cy;\n"
"mul.wide.u32 Ce0, pp0, iz0;\n"
"mul.wide.u32 Co0, pp0, iz1;\n"
"mul.wide.u32 Ce1, pp0, iz2;\n"
"mul.wide.u32 Co1, pp0, iz3;\n"
"mul.wide.u32 Ce2, pp0, iz4;\n"
"mul.wide.u32 Co2, pp0, iz5;\n"
"mul.wide.u32 Ce3, pp0, iz6;\n"
"mul.wide.u32 Co3, pp0, iz7;\n"
"mul.wide.u32 Ct, pp1, iz1;\n"
"mul.wide.u32 Codd_t, pp1, iz0;\n"
"add.cc.u64 Ce1, Ce1, Ct;\n"
"mul.wide.u32 Ct, pp1, iz3;\n"
"addc.cc.u64 Ce2, Ce2, Ct;\n"
"mul.wide.u32 Ct, pp1, iz5;\n"
"addc.cc.u64 Ce3, Ce3, Ct;\n"
"mul.wide.u32 Ct, pp1, iz7;\n"
"addc.u64 Ce4, Ct, 0;\n"
"add.cc.u64 Co0, Co0, Codd_t;\n"
"mul.wide.u32 Codd_t, pp1, iz2;\n"
"addc.cc.u64 Co1, Co1, Codd_t;\n"
"mul.wide.u32 Codd_t, pp1, iz4;\n"
"addc.cc.u64 Co2, Co2, Codd_t;\n"
"mul.wide.u32 Codd_t, pp1, iz6;\n"
"addc.cc.u64 Co3, Co3, Codd_t;\n"
"addc.u32 Codd_cy, 0, 0;\n"
"mul.wide.u32 Ct, pp2, iz0;\n"
"add.cc.u64 Ce1, Ce1, Ct;\n"
"mul.wide.u32 Ct, pp2, iz2;\n"
"addc.cc.u64 Ce2, Ce2, Ct;\n"
"mul.wide.u32 Ct, pp2, iz4;\n"
"addc.cc.u64 Ce3, Ce3, Ct;\n"
"mul.wide.u32 Ct, pp2, iz6;\n"
"addc.cc.u64 Ce4, Ce4, Ct;\n"
"addc.u32 Ccy, 0, 0;\n"
"mul.wide.u32 Codd_t, pp2, iz1;\n"
"mov.b64 Codd_lc, {Codd_cy, Ccy};\n"
"add.cc.u64 Co1, Co1, Codd_t;\n"
"mul.wide.u32 Codd_t, pp2, iz3;\n"
"addc.cc.u64 Co2, Co2, Codd_t;\n"
"mul.wide.u32 Codd_t, pp2, iz5;\n"
"addc.cc.u64 Co3, Co3, Codd_t;\n"
"mul.wide.u32 Codd_t, pp2, iz7;\n"
"addc.u64 Co4, Codd_t, Codd_lc;\n"
"mul.wide.u32 Ct, pp3, iz1;\n"
"mul.wide.u32 Codd_t, pp3, iz0;\n"
"add.cc.u64 Ce2, Ce2, Ct;\n"
"mul.wide.u32 Ct, pp3, iz3;\n"
"addc.cc.u64 Ce3, Ce3, Ct;\n"
"mul.wide.u32 Ct, pp3, iz5;\n"
"addc.cc.u64 Ce4, Ce4, Ct;\n"
"mul.wide.u32 Ct, pp3, iz7;\n"
"addc.u64 Ce5, Ct, 0;\n"
"add.cc.u64 Co1, Co1, Codd_t;\n"
"mul.wide.u32 Codd_t, pp3, iz2;\n"
"addc.cc.u64 Co2, Co2, Codd_t;\n"
"mul.wide.u32 Codd_t, pp3, iz4;\n"
"addc.cc.u64 Co3, Co3, Codd_t;\n"
"mul.wide.u32 Codd_t, pp3, iz6;\n"
"addc.cc.u64 Co4, Co4, Codd_t;\n"
"addc.u32 Codd_cy, 0, 0;\n"
"mul.wide.u32 Ct, pp4, iz0;\n"
"add.cc.u64 Ce2, Ce2, Ct;\n"
"mul.wide.u32 Ct, pp4, iz2;\n"
"addc.cc.u64 Ce3, Ce3, Ct;\n"
"mul.wide.u32 Ct, pp4, iz4;\n"
"addc.cc.u64 Ce4, Ce4, Ct;\n"
"mul.wide.u32 Ct, pp4, iz6;\n"
"addc.cc.u64 Ce5, Ce5, Ct;\n"
"addc.u32 Ccy, 0, 0;\n"
"mul.wide.u32 Codd_t, pp4, iz1;\n"
"mov.b64 Codd_lc, {Codd_cy, Ccy};\n"
"add.cc.u64 Co2, Co2, Codd_t;\n"
"mul.wide.u32 Codd_t, pp4, iz3;\n"
"addc.cc.u64 Co3, Co3, Codd_t;\n"
"mul.wide.u32 Codd_t, pp4, iz5;\n"
"addc.cc.u64 Co4, Co4, Codd_t;\n"
"mul.wide.u32 Codd_t, pp4, iz7;\n"
"addc.u64 Co5, Codd_t, Codd_lc;\n"
"mul.wide.u32 Ct, pp5, iz1;\n"
"mul.wide.u32 Codd_t, pp5, iz0;\n"
"add.cc.u64 Ce3, Ce3, Ct;\n"
"mul.wide.u32 Ct, pp5, iz3;\n"
"addc.cc.u64 Ce4, Ce4, Ct;\n"
"mul.wide.u32 Ct, pp5, iz5;\n"
"addc.cc.u64 Ce5, Ce5, Ct;\n"
"mul.wide.u32 Ct, pp5, iz7;\n"
"addc.u64 Ce6, Ct, 0;\n"
"add.cc.u64 Co2, Co2, Codd_t;\n"
"mul.wide.u32 Codd_t, pp5, iz2;\n"
"addc.cc.u64 Co3, Co3, Codd_t;\n"
"mul.wide.u32 Codd_t, pp5, iz4;\n"
"addc.cc.u64 Co4, Co4, Codd_t;\n"
"mul.wide.u32 Codd_t, pp5, iz6;\n"
"addc.cc.u64 Co5, Co5, Codd_t;\n"
"addc.u32 Codd_cy, 0, 0;\n"
"mul.wide.u32 Ct, pp6, iz0;\n"
"add.cc.u64 Ce3, Ce3, Ct;\n"
"mul.wide.u32 Ct, pp6, iz2;\n"
"addc.cc.u64 Ce4, Ce4, Ct;\n"
"mul.wide.u32 Ct, pp6, iz4;\n"
"addc.cc.u64 Ce5, Ce5, Ct;\n"
"mul.wide.u32 Ct, pp6, iz6;\n"
"addc.cc.u64 Ce6, Ce6, Ct;\n"
"addc.u32 Ccy, 0, 0;\n"
"mul.wide.u32 Codd_t, pp6, iz1;\n"
"mov.b64 Codd_lc, {Codd_cy, Ccy};\n"
"add.cc.u64 Co3, Co3, Codd_t;\n"
"mul.wide.u32 Codd_t, pp6, iz3;\n"
"addc.cc.u64 Co4, Co4, Codd_t;\n"
"mul.wide.u32 Codd_t, pp6, iz5;\n"
"addc.cc.u64 Co5, Co5, Codd_t;\n"
"mul.wide.u32 Codd_t, pp6, iz7;\n"
"addc.u64 Co6, Codd_t, Codd_lc;\n"
"mul.wide.u32 Ct, pp7, iz1;\n"
"mul.wide.u32 Codd_t, pp7, iz0;\n"
"add.cc.u64 Ce4, Ce4, Ct;\n"
"mul.wide.u32 Ct, pp7, iz3;\n"
"addc.cc.u64 Ce5, Ce5, Ct;\n"
"mul.wide.u32 Ct, pp7, iz5;\n"
"addc.cc.u64 Ce6, Ce6, Ct;\n"
"mul.wide.u32 Ct, pp7, iz7;\n"
"addc.u64 Ce7, Ct, 0;\n"
"add.cc.u64 Co3, Co3, Codd_t;\n"
"mul.wide.u32 Codd_t, pp7, iz2;\n"
"addc.cc.u64 Co4, Co4, Codd_t;\n"
"mul.wide.u32 Codd_t, pp7, iz4;\n"
"addc.cc.u64 Co5, Co5, Codd_t;\n"
"mul.wide.u32 Codd_t, pp7, iz6;\n"
"addc.cc.u64 Co6, Co6, Codd_t;\n"
"mov.b64 {Cx14,Cx15}, Ce7; addc.u32 Cx15, Cx15, 0;\n"
"mov.b64 {Cx0,Cx1}, Ce0;\n"
"	mov.b64 {Cx2,Cx3}, Ce1;\n"
"	mov.b64 {Cx4,Cx5}, Ce2;\n"
"	mov.b64 {Cx6,Cx7}, Ce3;\n"
"	mov.b64 {Cx8,Cx9}, Ce4;\n"
"	mov.b64 {Cx10,Cx11}, Ce5;\n"
"	mov.b64 {Cx12,Cx13}, Ce6;\n"
"	\n"
"	mov.b64 {Cy1,Cy2}, Co0;\n"
"	mov.b64 {Cy3,Cy4}, Co1;\n"
"	mov.b64 {Cy5,Cy6}, Co2;\n"
"	mov.b64 {Cy7,Cy8}, Co3;\n"
"	mov.b64 {Cy9,Cy10}, Co4;\n"
"	mov.b64 {Cy11,Cy12}, Co5;\n"
"	mov.b64 {Cy13,Cy14}, Co6;\n"
"	add.cc.u32 Cx1, Cx1, Cy1;\n"
"	addc.cc.u32 Cx2, Cx2, Cy2;\n"
"	addc.cc.u32 Cx3, Cx3, Cy3;\n"
"	addc.cc.u32 Cx4, Cx4, Cy4;\n"
"	addc.cc.u32 Cx5, Cx5, Cy5;\n"
"	addc.cc.u32 Cx6, Cx6, Cy6;\n"
"	addc.cc.u32 Cx7, Cx7, Cy7;\n"
"	addc.cc.u32 Cx8, Cx8, Cy8;\n"
"	addc.cc.u32 Cx9, Cx9, Cy9;\n"
"	addc.cc.u32 Cx10, Cx10, Cy10;\n"
"	addc.cc.u32 Cx11, Cx11, Cy11;\n"
"	addc.cc.u32 Cx12, Cx12, Cy12;\n"
"	addc.cc.u32 Cx13, Cx13, Cy13;\n"
"	addc.cc.u32 Cx14, Cx14, Cy14;\n"
"	addc.u32 Cx15, Cx15, 0;\n"
"	.reg .u64 Cr0,Cr1,Cr2,Cr3,Ch0,Ch1,Ch2,Ch3,Cf0,Cf1,Cf2,Cf3,Cg0,Cg1,Cg2,Cg3;\n"
"	.reg .u32 Cf8,Cg8,Cz8,Cz9,Cw0,Cw1,Cw2,Cw3,Cw4,Cw5,Cw6,Cw7,Cm0,Cm1,Cm2;\n"
"	mov.b64 Cr0, {Cx0,Cx1}; mov.b64 Cr1, {Cx2,Cx3}; mov.b64 Cr2, {Cx4,Cx5}; mov.b64 Cr3, {Cx6,Cx7};\n"
"	mov.b64 Ch0, {Cx8,Cx9}; mov.b64 Ch1, {Cx10,Cx11}; mov.b64 Ch2, {Cx12,Cx13}; mov.b64 Ch3, {Cx14,Cx15};\n"
"	mul.wide.u32 Ct, Cx8, 977;  add.cc.u64  Cf0, Cr0, Ct;\n"
"	mul.wide.u32 Ct, Cx10, 977; addc.cc.u64 Cf1, Cr1, Ct;\n"
"	mul.wide.u32 Ct, Cx12, 977; addc.cc.u64 Cf2, Cr2, Ct;\n"
"	mul.wide.u32 Ct, Cx14, 977; addc.cc.u64 Cf3, Cr3, Ct;\n"
"	mul.wide.u32 Ct, Cx9, 977;  add.cc.u64  Cg0, Ch0, Ct;\n"
"	mul.wide.u32 Ct, Cx11, 977; addc.cc.u64 Cg1, Ch1, Ct;\n"
"	mul.wide.u32 Ct, Cx13, 977; addc.cc.u64 Cg2, Ch2, Ct;\n"
"	mul.wide.u32 Ct, Cx15, 977; addc.cc.u64 Cg3, Ch3, Ct;\n"
"	/*rp*/\n"
"	mov.b64 {oz0,oz1}, Cf0;\n"
"	mov.b64 {oz2,oz3}, Cf1;\n"
"	mov.b64 {oz4,oz5}, Cf2;\n"
"	mov.b64 {oz6,oz7}, Cf3;\n"
"	mov.b64 {Cw0,Cw1}, Cg0;\n"
"	mov.b64 {Cw2,Cw3}, Cg1;\n"
"	mov.b64 {Cw4,Cw5}, Cg2;\n"
"	mov.b64 {Cw6,Cw7}, Cg3;\n"
"	add.cc.u32  oz1, oz1, Cw0;\n"
"	addc.cc.u32 oz2, oz2, Cw1;\n"
"	addc.cc.u32 oz3, oz3, Cw2;\n"
"	addc.cc.u32 oz4, oz4, Cw3;\n"
"	addc.cc.u32 oz5, oz5, Cw4;\n"
"	addc.cc.u32 oz6, oz6, Cw5;\n"
"	addc.cc.u32 oz7, oz7, Cw6;\n"
"	addc.u32 Cz8, 0, Cw7;\n"
"	{ .reg .u64 Csfa,Csft; .reg .u32 Csfl,Csfh;\n"
"mul.wide.u32 Csft, Cz8, 977;\n"
"mov.b64 Csfa, {oz0, oz1}; add.cc.u64 Csfa, Csfa, Csft; addc.u32 oz2, oz2, 0; mov.b64 {Csfl, Csfh}, Csfa;\n"
"\n"
"\n"
"mov.u32 oz0, Csfl;\n"
"add.cc.u32 oz1, Csfh, Cz8; addc.u32 oz2, oz2, 0;\n"
" }\n"
"\n"
"\n"
"	mov.b64 %0, {op0,op1};\n"
"	mov.b64 %1, {op2,op3};\n"
"	mov.b64 %2, {op4,op5};\n"
"	mov.b64 %3, {op6,op7};\n"
"	mov.b64 %4, {oq0,oq1};\n"
"	mov.b64 %5, {oq2,oq3};\n"
"	mov.b64 %6, {oq4,oq5};\n"
"	mov.b64 %7, {oq6,oq7};\n"
"	mov.b64 %8, {oz0,oz1};\n"
"	mov.b64 %9, {oz2,oz3};\n"
"	mov.b64 %10, {oz4,oz5};\n"
"	mov.b64 %11, {oz6,oz7};\n"
"}\n"
"\n"
        : "=l"(p0),"=l"(p1),"=l"(p2),"=l"(p3),
          "=l"(q0),"=l"(q1),"=l"(q2),"=l"(q3),
          "=l"(z0),"=l"(z1),"=l"(z2),"=l"(z3)
        : "l"(U2[0]),"l"(U2[1]),"l"(U2[2]),"l"(U2[3]),
          "l"(X1[0]),"l"(X1[1]),"l"(X1[2]),"l"(X1[3]),
          "l"(ZZ[0]),"l"(ZZ[1]),"l"(ZZ[2]),"l"(ZZ[3]));
    PPP[0]=p0; PPP[1]=p1; PPP[2]=p2; PPP[3]=p3;
    Q[0]=q0; Q[1]=q1; Q[2]=q2; Q[3]=q3;
    ZZ[0]=z0; ZZ[1]=z1; ZZ[2]=z2; ZZ[3]=z3;
#else
    uint64_t P[4], PP[4];
    _ModSub256C(P, U2, X1);
    _ModSqr(PP, P);
    _ModMult(PPP, PP, P);
    _ModMult(Q, (uint64_t *)U2, PP);
    _ModMult(ZZ, PP);
#endif
}
#endif

#if QSB_U32_TRIP == 4

__device__ __forceinline__ void qsb_mul4(uint32_t r[8], const uint32_t a[4], const uint32_t b[4]) {
    unsigned __int128 col[7];
#pragma unroll
    for (int i = 0; i < 7; i++) col[i] = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            uint64_t p;
            asm("mul.wide.u32 %0, %1, %2;" : "=l"(p) : "r"(a[i]), "r"(b[j]));
            col[i + j] += p;
        }
    }
    unsigned __int128 cy = 0;
#pragma unroll
    for (int i = 0; i < 7; i++) {
        cy += col[i];
        r[i] = (uint32_t)cy;
        cy >>= 32;
    }
    r[7] = (uint32_t)cy;
}

__device__ __forceinline__ void qsb_mul128(uint64_t r[4], uint64_t a0, uint64_t a1,
                                           uint64_t b0, uint64_t b1) {
    uint64_t p00l, p00h, p01l, p01h, p10l, p10h, p11l, p11h;
    asm("mul.lo.u64 %0, %2, %3;\n\tmul.hi.u64 %1, %2, %3;"
        : "=l"(p00l), "=l"(p00h) : "l"(a0), "l"(b0));
    asm("mul.lo.u64 %0, %2, %3;\n\tmul.hi.u64 %1, %2, %3;"
        : "=l"(p01l), "=l"(p01h) : "l"(a0), "l"(b1));
    asm("mul.lo.u64 %0, %2, %3;\n\tmul.hi.u64 %1, %2, %3;"
        : "=l"(p10l), "=l"(p10h) : "l"(a1), "l"(b0));
    asm("mul.lo.u64 %0, %2, %3;\n\tmul.hi.u64 %1, %2, %3;"
        : "=l"(p11l), "=l"(p11h) : "l"(a1), "l"(b1));
    r[0] = p00l;
    unsigned __int128 s = (unsigned __int128)p00h + p01l + p10l;
    r[1] = (uint64_t)s;
    s = (unsigned __int128)(uint64_t)(s >> 64) + p01h + p10h + p11l;
    r[2] = (uint64_t)s;
    r[3] = (uint64_t)(s >> 64) + p11h;
}

__device__ __forceinline__ void qsb_kadd_at(uint64_t *a, int n, int idx, uint64_t val) {
    unsigned __int128 v = val;
    for (int i = idx; v && i < n; i++) {
        v += a[i];
        a[i] = (uint64_t)v;
        v >>= 64;
    }
}

__device__ __forceinline__ void qsb_ksub_limbs(uint64_t *a, int n, const uint64_t *b, int nb) {
    uint64_t br = 0;
    int i = 0;
    for (; i < nb; i++) {
        unsigned __int128 v = (unsigned __int128)a[i] - b[i] - br;
        a[i] = (uint64_t)v;
        br = (uint64_t)(v >> 127);
    }
    for (; br && i < n; i++) {
        unsigned __int128 v = (unsigned __int128)a[i] - br;
        a[i] = (uint64_t)v;
        br = (uint64_t)(v >> 127);
    }
}

__device__ __forceinline__ void qsb_kadd_limbs(uint64_t *d, int n, int idx, const uint64_t *s, int ns) {
    uint64_t c = 0;
    int j = 0;
    for (; j < ns && idx + j < n; j++) {
        unsigned __int128 v = (unsigned __int128)d[idx + j] + s[j] + c;
        d[idx + j] = (uint64_t)v;
        c = (uint64_t)(v >> 64);
    }
    for (int i = idx + j; c && i < n; i++) {
        unsigned __int128 v = (unsigned __int128)d[i] + c;
        d[i] = (uint64_t)v;
        c = (uint64_t)(v >> 64);
    }
}

__device__ __forceinline__ void qsb_kmul(uint64_t r[4], const uint64_t *a, const uint64_t *b) {
    uint32_t A[8], B[8], z0[8], z2[8], s[4], t[4], st[8], prod[16];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        A[2 * i] = (uint32_t)a[i]; A[2 * i + 1] = (uint32_t)(a[i] >> 32);
        B[2 * i] = (uint32_t)b[i]; B[2 * i + 1] = (uint32_t)(b[i] >> 32);
    }
    qsb_mul4(z0, A, B);
    qsb_mul4(z2, A + 4, B + 4);
    uint32_t sc, tc;
    {
        uint64_t c = 0;
#pragma unroll
        for (int i = 0; i < 4; i++) { c += (uint64_t)A[i] + A[4 + i]; s[i] = (uint32_t)c; c >>= 32; }
        sc = (uint32_t)c;
        c = 0;
#pragma unroll
        for (int i = 0; i < 4; i++) { c += (uint64_t)B[i] + B[4 + i]; t[i] = (uint32_t)c; c >>= 32; }
        tc = (uint32_t)c;
    }
    qsb_mul4(st, s, t);
#pragma unroll
    for (int i = 0; i < 16; i++) prod[i] = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) prod[i] = z0[i];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        uint64_t v = (uint64_t)prod[8 + i] + z2[i];
        prod[8 + i] = (uint32_t)v;
        uint32_t cy = (uint32_t)(v >> 32);
        for (int k = 9 + i; cy && k < 16; k++) {
            uint64_t w = (uint64_t)prod[k] + cy;
            prod[k] = (uint32_t)w;
            cy = (uint32_t)(w >> 32);
        }
    }

    uint32_t mid[12];
#pragma unroll
    for (int i = 0; i < 12; i++) mid[i] = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) mid[i] = st[i];
    if (sc) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            uint64_t v = (uint64_t)mid[4 + i] + t[i];
            mid[4 + i] = (uint32_t)v;
            uint32_t cy = (uint32_t)(v >> 32);
            for (int k = 5 + i; cy && k < 12; k++) {
                uint64_t w = (uint64_t)mid[k] + cy; mid[k] = (uint32_t)w; cy = (uint32_t)(w >> 32);
            }
        }
    }
    if (tc) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            uint64_t v = (uint64_t)mid[4 + i] + s[i];
            mid[4 + i] = (uint32_t)v;
            uint32_t cy = (uint32_t)(v >> 32);
            for (int k = 5 + i; cy && k < 12; k++) {
                uint64_t w = (uint64_t)mid[k] + cy; mid[k] = (uint32_t)w; cy = (uint32_t)(w >> 32);
            }
        }
    }
    if (sc & tc) {
        uint64_t v = (uint64_t)mid[8] + 1;
        mid[8] = (uint32_t)v;
        mid[9] += (uint32_t)(v >> 32);
    }
    {
        uint32_t br = 0;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            uint32_t v = mid[i] - z0[i];
            uint32_t br1 = mid[i] < z0[i];
            mid[i] = v - br;
            br = br1 || (v < br);
        }
#pragma unroll
        for (int i = 8; i < 12; i++) {
            uint32_t v = mid[i];
            mid[i] = v - br;
            br = v < br;
        }
        br = 0;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            uint32_t v = mid[i] - z2[i];
            uint32_t br1 = mid[i] < z2[i];
            mid[i] = v - br;
            br = br1 || (v < br);
        }
#pragma unroll
        for (int i = 8; i < 12; i++) {
            uint32_t v = mid[i];
            mid[i] = v - br;
            br = v < br;
        }
    }
    {
        uint32_t cy = 0;
#pragma unroll
        for (int i = 0; i < 12; i++) {
            uint64_t v = (uint64_t)prod[4 + i] + mid[i] + cy;
            prod[4 + i] = (uint32_t)v;
            cy = (uint32_t)(v >> 32);
        }
    }

#pragma unroll
    for (int fold = 0; fold < 3; fold++) {
        uint32_t hi[8];
#pragma unroll
        for (int i = 0; i < 8; i++) { hi[i] = prod[8 + i]; prod[8 + i] = 0; }
        uint32_t cy = 0;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            uint64_t w = (uint64_t)hi[i] * 977u + cy;
            if (i) w += (uint64_t)hi[i - 1];

            w += prod[i];
            prod[i] = (uint32_t)w;
            cy = (uint32_t)(w >> 32);
        }
        prod[8] = cy;
#pragma unroll
        for (int i = 0; i < 7; i++) prod[9 + i] = 0;
        (void)hi;
    }
#pragma unroll
    for (int i = 0; i < 4; i++)
        r[i] = (uint64_t)prod[2 * i] | ((uint64_t)prod[2 * i + 1] << 32);
}
#if 0
    qsb_mul128(z2, a[2], a[3], b[2], b[3]);
    unsigned __int128 ssum = (unsigned __int128)a[0] + a[2];
    uint64_t s0 = (uint64_t)ssum;
    ssum = (ssum >> 64) + a[1] + a[3];
    uint64_t s1 = (uint64_t)ssum;
    uint64_t sc = (uint64_t)(ssum >> 64);
    unsigned __int128 tsum = (unsigned __int128)b[0] + b[2];
    uint64_t t0 = (uint64_t)tsum;
    tsum = (tsum >> 64) + b[1] + b[3];
    uint64_t t1 = (uint64_t)tsum;
    uint64_t tc = (uint64_t)(tsum >> 64);
    qsb_mul128(st, s0, s1, t0, t1);
#pragma unroll
    for (int i = 0; i < 6; i++) m[i] = 0;
    m[0] = st[0]; m[1] = st[1]; m[2] = st[2]; m[3] = st[3];
    if (sc) { qsb_kadd_at(m, 6, 2, t0); qsb_kadd_at(m, 6, 3, t1); }
    if (tc) { qsb_kadd_at(m, 6, 2, s0); qsb_kadd_at(m, 6, 3, s1); }
    if (sc & tc) qsb_kadd_at(m, 6, 4, 1);
    qsb_ksub_limbs(m, 6, z0, 4);
    qsb_ksub_limbs(m, 6, z2, 4);
#pragma unroll
    for (int i = 0; i < 8; i++) prod[i] = 0;
    prod[0] = z0[0];
    prod[1] = z0[1];
    qsb_kadd_limbs(prod, 8, 2, z0 + 2, 2);
    qsb_kadd_limbs(prod, 8, 2, m, 6);
    qsb_kadd_limbs(prod, 8, 4, z2, 4);

#pragma unroll
    for (int fold = 0; fold < 3; fold++) {
        uint64_t hi0 = prod[4], hi1 = prod[5], hi2 = prod[6], hi3 = prod[7];
        prod[4] = prod[5] = prod[6] = prod[7] = 0;

        uint64_t h32_0 = hi0 << 32;
        uint64_t h32_1 = (hi0 >> 32) | (hi1 << 32);
        uint64_t h32_2 = (hi1 >> 32) | (hi2 << 32);
        uint64_t h32_3 = (hi2 >> 32) | (hi3 << 32);
        uint64_t h32_4 = hi3 >> 32;
        uint64_t k0, k1, k2, k3, k4;
        asm("mul.lo.u64 %0, %1, 977;" : "=l"(k0) : "l"(hi0));
        asm("mul.hi.u64 %0, %1, 977;" : "=l"(k1) : "l"(hi0));
        uint64_t tlo, thi;
        asm("mul.lo.u64 %0, %2, 977;\n\tmul.hi.u64 %1, %2, 977;"
            : "=l"(tlo), "=l"(thi) : "l"(hi1));
        unsigned __int128 acc = (unsigned __int128)k1 + tlo;
        k1 = (uint64_t)acc;
        acc = (acc >> 64) + thi;
        asm("mul.lo.u64 %0, %2, 977;\n\tmul.hi.u64 %1, %2, 977;"
            : "=l"(tlo), "=l"(thi) : "l"(hi2));
        acc += tlo;
        k2 = (uint64_t)acc;
        acc = (acc >> 64) + thi;
        asm("mul.lo.u64 %0, %2, 977;\n\tmul.hi.u64 %1, %2, 977;"
            : "=l"(tlo), "=l"(thi) : "l"(hi3));
        acc += tlo;
        k3 = (uint64_t)acc;
        k4 = (uint64_t)(acc >> 64) + thi;
        unsigned __int128 w = (unsigned __int128)prod[0] + h32_0 + k0;
        prod[0] = (uint64_t)w;
        w = (w >> 64) + prod[1] + h32_1 + k1;
        prod[1] = (uint64_t)w;
        w = (w >> 64) + prod[2] + h32_2 + k2;
        prod[2] = (uint64_t)w;
        w = (w >> 64) + prod[3] + h32_3 + k3;
        prod[3] = (uint64_t)w;
        w = (w >> 64) + h32_4 + k4;
        prod[4] = (uint64_t)w;
        prod[5] = (uint64_t)(w >> 64);
    }
    r[0] = prod[0]; r[1] = prod[1]; r[2] = prod[2]; r[3] = prod[3];
}
#endif
#endif
