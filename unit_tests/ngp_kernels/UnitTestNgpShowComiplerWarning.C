/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "gtest/gtest.h"
#include "UnitTestUtils.h"
#include <NaluEnv.h>

#include "ngp_utils/NgpTypes.h"
#include "ngp_utils/NgpLoopUtils.h"

namespace sierra{
namespace nalu{
namespace nalu_ngp {

struct BaseClass
{
  KOKKOS_FUNCTION virtual ~BaseClass() {} 
  KOKKOS_FUNCTION virtual void Warning(int,
    SharedMemView<DoubleType**, DeviceShmem>&,
    SharedMemView<DoubleType***, DeviceShmem>&) { }
  KOKKOS_FUNCTION virtual void NoWarning(
    SharedMemView<DoubleType**, DeviceShmem>&,
    SharedMemView<DoubleType***, DeviceShmem>&,   
    SharedMemView<DoubleType***, DeviceShmem>&) { }
};
struct DerivedClass : public BaseClass {
  KOKKOS_FUNCTION virtual ~DerivedClass() {} 
};

class NgpCompileTest : public ::testing::Test {};

void show_cuda_compiler_warning()
{
  auto derived = sierra::nalu::create_device_expression<DerivedClass>();
  using ShmemType = typename NGPMeshTraits<ngp::Mesh>::ShmemType;
  Kokkos::parallel_for(1, KOKKOS_LAMBDA(const size_t&i) {
    SharedMemView<DoubleType**,  ShmemType> D2;
    SharedMemView<DoubleType***, ShmemType> D3;
    if (!i) {
#define TRIGGER_WARNING
#ifdef TRIGGER_WARNING
      derived->Warning(0, D2, D3);
#endif
      derived->NoWarning(D2, D3, D3);
    }
  });
}

TEST_F(NgpCompileTest, Show_Compiler_Warning)
{
  show_cuda_compiler_warning();
}

}}}
