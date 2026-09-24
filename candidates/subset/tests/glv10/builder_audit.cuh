static void audit_builder(const uint8_t nri[32]) {
 std::vector<uint32_t> indices;
 for(int c=0;c<5;c++){
  for(unsigned i=0;i<=256;i++)indices.push_back(glv10_offset(c)+i);
  unsigned edges[]={4094,4095,4096,4097,8190,8191,8192,8193,glv10_entries(c)-2,glv10_entries(c)-1};
  for(unsigned i:edges)indices.push_back(glv10_offset(c)+i);
 }
 const unsigned count=indices.size();must(count==1335&&count%256==55,"partial CTA domain");
 size_t words=5*GLV10_LADDER*8,bytes=words*8;std::vector<uint64_t>L(words),H(words);
 glv10_build_ladders(L.data(),H.data(),nri);
 uint64_t *dl,*dh;uint32_t *di;unsigned *failed;uint8_t *reference,*batch;
 qsb_native::check(cudaMalloc(&dl,bytes),"builder audit L");qsb_native::check(cudaMalloc(&dh,bytes),"builder audit H");
 qsb_native::check(cudaMalloc(&di,count*4),"builder audit indices");qsb_native::check(cudaMalloc(&failed,4),"builder audit error");
 qsb_native::check(cudaMalloc(&reference,(size_t)count*64),"builder audit reference");qsb_native::check(cudaMalloc(&batch,(size_t)count*64),"builder audit batch");
 qsb_native::check(cudaMemcpy(dl,L.data(),bytes,cudaMemcpyHostToDevice),"builder audit L upload");qsb_native::check(cudaMemcpy(dh,H.data(),bytes,cudaMemcpyHostToDevice),"builder audit H upload");
 qsb_native::check(cudaMemcpy(di,indices.data(),count*4,cudaMemcpyHostToDevice),"builder audit index upload");qsb_native::check(cudaMemset(failed,0,4),"builder audit error reset");
 kernel_build_glv10<<<(count+255)/256,256>>>(dl,dh,reference,0,count,di);
 kernel_build_glv10_batch<<<(count+255)/256,256>>>(dl,dh,batch,0,count,di,failed);
 qsb_native::check(cudaDeviceSynchronize(),"builder audit kernels");
 unsigned error;std::vector<uint64_t>a((size_t)count*8),b((size_t)count*8);
 qsb_native::check(cudaMemcpy(&error,failed,4,cudaMemcpyDeviceToHost),"builder audit error read");must(!error,"builder unexpected zero denominator");
 qsb_native::check(cudaMemcpy(a.data(),reference,(size_t)count*64,cudaMemcpyDeviceToHost),"builder audit reference read");qsb_native::check(cudaMemcpy(b.data(),batch,(size_t)count*64,cudaMemcpyDeviceToHost),"builder audit batch read");
 EC_GROUP*g=EC_GROUP_new_by_curve_name(NID_secp256k1);BN_CTX*ctx=BN_CTX_new();EC_POINT*point=EC_POINT_new(g);
 BIGNUM *alpha=BN_lebin2bn(nri,32,NULL),*k=BN_new(),*tmp=BN_new(),*x=BN_new(),*y=BN_new();
 for(unsigned i=0;i<count;i++){
  int c=4;while(c&&indices[i]<glv10_offset(c))c--;
  glv10_scalar(k,c,indices[i]-glv10_offset(c),tmp);must(BN_mul(k,k,alpha,ctx)&&EC_POINT_mul(g,point,k,NULL,NULL,ctx)&&EC_POINT_get_affine_coordinates(g,point,x,y,ctx),"builder oracle");
  uint64_t want[8];BN_bn2lebinpad(x,(unsigned char*)want,32);BN_bn2lebinpad(y,(unsigned char*)(want+4),32);
  if(memcmp(want,a.data()+8*i,64)||memcmp(want,b.data()+8*i,64)){fprintf(stderr,"builder audit mismatch sample=%u table_index=%u\n",i,indices[i]);exit(4);}
 }
 printf("builder audit records=%u segments=5 exhaustive_indices=0..256 tail_live=55 tail_padded=201 PASS\n",count);
 cudaFree(dl);cudaFree(dh);cudaFree(di);cudaFree(failed);cudaFree(reference);cudaFree(batch);
 BN_free(alpha);BN_free(k);BN_free(tmp);BN_free(x);BN_free(y);EC_POINT_free(point);EC_GROUP_free(g);BN_CTX_free(ctx);
}
