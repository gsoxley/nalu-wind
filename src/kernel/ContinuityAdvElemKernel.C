/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernel/ContinuityAdvElemKernel.h"
#include "AlgTraits.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementFactory.h"
#include "TimeIntegrator.h"
#include "SolutionOptions.h"

// template and scratch space
#include "BuildTemplates.h"
#include "ScratchViews.h"
#include "utils/StkHelpers.h"

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>

namespace sierra {
namespace nalu {

template<typename AlgTraits>
ContinuityAdvElemKernel<AlgTraits>::ContinuityAdvElemKernel(
  const stk::mesh::BulkData& bulkData,
  const SolutionOptions& solnOpts,
  ElemDataRequests& dataPreReqs)
  : meshMotion_(solnOpts.does_mesh_move()),
    shiftMdot_(solnOpts.cvfemShiftMdot_),
    shiftPoisson_(solnOpts.get_shifted_grad_op("pressure")),
    reducedSensitivities_(solnOpts.cvfemReducedSensPoisson_),
    interpTogether_(solnOpts.get_mdot_interp()),
    om_interpTogether_(1.0 - interpTogether_)
{
  // Save of required fields
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();
  std::string velocity_name = meshMotion_ ? "velocity_rtm" : "velocity";

  velocityRTM_ = get_field_ordinal(metaData, velocity_name);
  Gpdx_ = get_field_ordinal(metaData, "dpdx");
  pressure_ = get_field_ordinal(metaData, "pressure");
  densityNp1_ = get_field_ordinal(metaData, "density", stk::mesh::StateNP1);
  coordinates_ = get_field_ordinal(metaData, solnOpts.get_coordinates_name());
  Udiag_ = get_field_ordinal(metaData, "momentum_diag");

  meSCS_ = sierra::nalu::MasterElementRepo::get_surface_master_element<AlgTraits>();
  dataPreReqs.add_cvfem_surface_me(meSCS_);

  // fields and data
  dataPreReqs.add_coordinates_field(coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataPreReqs.add_gathered_nodal_field(velocityRTM_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(densityNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(pressure_, 1);
  dataPreReqs.add_gathered_nodal_field(Udiag_, 1);
  dataPreReqs.add_gathered_nodal_field(Gpdx_, AlgTraits::nDim_);
  dataPreReqs.add_master_element_call(SCS_AREAV, CURRENT_COORDINATES);

  // manage dndx
  if ( !shiftPoisson_ || !reducedSensitivities_ )
    dataPreReqs.add_master_element_call(SCS_GRAD_OP, CURRENT_COORDINATES);
  if ( shiftPoisson_ || reducedSensitivities_ )
    dataPreReqs.add_master_element_call(SCS_SHIFTED_GRAD_OP, CURRENT_COORDINATES);

  dataPreReqs.add_master_element_call(
    (shiftMdot_ ? SCS_SHIFTED_SHAPE_FCN : SCS_SHAPE_FCN), CURRENT_COORDINATES);
}

template<typename AlgTraits>
void
ContinuityAdvElemKernel<AlgTraits>::setup(const TimeIntegrator& timeIntegrator)
{
  const double dt = timeIntegrator.get_time_step();
  const double gamma1 = timeIntegrator.get_gamma1();
  projTimeScale_ = dt / gamma1;
}

template<typename AlgTraits>
void
ContinuityAdvElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**, DeviceShmem>& lhs,
  SharedMemView<DoubleType*, DeviceShmem>& rhs,
  ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>& scratchViews)
{
  // Work arrays (fixed size)
  NALU_ALIGNED DoubleType w_uIp     [AlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_rho_uIp [AlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_Gpdx_Ip [AlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_dpdxIp  [AlgTraits::nDim_];

  auto& v_densityNp1 = scratchViews.get_scratch_view_1D(densityNp1_);
  auto& v_pressure = scratchViews.get_scratch_view_1D(pressure_);
  auto& v_udiag = scratchViews.get_scratch_view_1D(Udiag_);

  auto& v_velocity = scratchViews.get_scratch_view_2D(velocityRTM_);
  auto& v_Gpdx = scratchViews.get_scratch_view_2D(Gpdx_);

  auto& meViews = scratchViews.get_me_views(CURRENT_COORDINATES);
  auto& v_scs_areav = meViews.scs_areav;

  auto& v_dndx = shiftPoisson_ ? meViews.dndx_shifted : meViews.dndx;
  auto& v_dndx_lhs = (shiftPoisson_ || reducedSensitivities_)? meViews.dndx_shifted : meViews.dndx;
  auto& v_shape_function = shiftMdot_ ? meViews.scs_shifted_shape_fcn : meViews.scs_shape_fcn;

  const int* lrscv = meSCS_->adjacentNodes();

  for (int ip = 0; ip < AlgTraits::numScsIp_; ++ip) {
    const int il = lrscv[2*ip];
    const int ir = lrscv[2*ip+1];

    DoubleType rhoIp = 0.0;
    DoubleType projTimeScaleIP = 0.0;
    for (int j = 0; j < AlgTraits::nDim_; ++j) {
      w_uIp[j] = 0.0;
      w_rho_uIp[j] = 0.0;
      w_Gpdx_Ip[j] = 0.0;
      w_dpdxIp[j] = 0.0;
    }

    for (int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic) {
      const DoubleType r = v_shape_function(ip, ic);
      projTimeScaleIP += r / v_udiag(ic);
    }

    for (int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic) {
      const DoubleType r = v_shape_function(ip, ic);
      const DoubleType nodalPressure = v_pressure(ic);
      const DoubleType nodalRho = v_densityNp1(ic);
      const DoubleType udiagInv = 1.0 / v_udiag(ic);

      rhoIp += r * nodalRho;

      DoubleType lhsfac = 0.0;
      for (int j = 0; j < AlgTraits::nDim_; ++j) {
        w_Gpdx_Ip[j] += r * v_Gpdx(ic, j) * udiagInv;
        w_uIp[j]     += r * v_velocity(ic, j);
        w_rho_uIp[j] += r * nodalRho * v_velocity(ic, j);
        w_dpdxIp[j]  += v_dndx(ip, ic, j) * nodalPressure;
        lhsfac += -v_dndx_lhs(ip, ic, j) * v_scs_areav(ip, j) * projTimeScaleIP;
      }

      lhs(il,ic) += lhsfac / projTimeScale_;
      lhs(ir,ic) -= lhsfac / projTimeScale_;
    }

    // assemble mdot
    DoubleType mdot = 0.0;
    for (int j = 0; j < AlgTraits::nDim_; ++j) {
      mdot += (interpTogether_ * w_rho_uIp[j] + om_interpTogether_ * rhoIp * w_uIp[j] -
               ( projTimeScaleIP * w_dpdxIp[j] - w_Gpdx_Ip[j])) * v_scs_areav(ip,j);
    }

    // residuals
    rhs(il) -= mdot / projTimeScale_;
    rhs(ir) += mdot / projTimeScale_;
  }
}

INSTANTIATE_KERNEL(ContinuityAdvElemKernel)

}  // nalu
}  // sierra
