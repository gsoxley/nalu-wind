/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "ngp_algorithms/EffSSTDiffFluxCoeffAlg.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpTypes.h"
#include "Realm.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/MetaData.hpp"

namespace sierra{
namespace nalu{

EffSSTDiffFluxCoeffAlg::EffSSTDiffFluxCoeffAlg(
  Realm &realm,
  stk::mesh::Part *part,
  ScalarFieldType *visc,
  ScalarFieldType *tvisc,
  ScalarFieldType *evisc,
  const double sigmaOne,
  const double sigmaTwo)
  : Algorithm(realm, part),
    viscField_(visc),
    visc_(visc->mesh_meta_data_ordinal()),
    tvisc_(tvisc->mesh_meta_data_ordinal()),
    evisc_(evisc->mesh_meta_data_ordinal()),
    fOneBlend_(get_field_ordinal(realm.meta_data(), "sst_f_one_blending")),
    sigmaOne_(sigmaOne),
    sigmaTwo_(sigmaTwo)
{}

void
EffSSTDiffFluxCoeffAlg::execute()
{
  using Traits = nalu_ngp::NGPMeshTraits<ngp::Mesh>;

  const auto& meta = realm_.meta_data();

  stk::mesh::Selector sel
    = (meta.locally_owned_part() | meta.globally_shared_part())
    &stk::mesh::selectField(*viscField_);

  const auto& meshInfo = realm_.mesh_info();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  const auto visc = fieldMgr.get_field<double>(visc_);
  const auto tvisc = fieldMgr.get_field<double>(tvisc_);
  auto evisc = fieldMgr.get_field<double>(evisc_);
  const auto fOneBlend = fieldMgr.get_field<double>(fOneBlend_);

  const DblType sigmaOne = sigmaOne_;
  const DblType sigmaTwo = sigmaTwo_;

  nalu_ngp::run_entity_algorithm(
    ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const Traits::MeshIndex& meshIdx) {
      const DblType blendedConstant = fOneBlend.get(meshIdx, 0)*sigmaOne + (1.0-fOneBlend.get(meshIdx, 0))*sigmaTwo;
      evisc.get(meshIdx, 0) = visc.get(meshIdx, 0) + tvisc.get(meshIdx, 0) * blendedConstant;
    });

  // Set flag indicating that the field has been modified on device
  evisc.modify_on_device();
}

} // namespace nalu
} // namespace Sierra
