#pragma once
#include "Dynamics/ParticleSystem/HyperelasticBody.h"

namespace PhysIKA
{
	template<typename> class NeighborQuery;
	template<typename> class PointSetToPointSet;
	template<typename> class ParticleIntegrator;
	template<typename> class ElasticityModule;
	template<typename> class HyperelastoplasticityModule;
	template<typename> class HyperelasticFractureModule;
	template<typename> class DensityPBD;
	template<typename TDataType> class ImplicitViscosity;
	/*!
	*	\class	HyperelastoplasticityBody
	*	\brief	Peridynamics-based elastoplastic object.
	*/
	template<typename TDataType>
	class HyperelastoplasticityBody : public HyperelasticBody<TDataType>
	{
		DECLARE_CLASS_1(HyperelastoplasticityBody, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		HyperelastoplasticityBody(std::string name = "default");
		virtual ~HyperelastoplasticityBody();

		void advance(Real dt) override;

		void updateTopology() override;

		bool initialize() override;

		bool translate(Coord t) override;
		bool scale(Real s) override;

		void loadSurface(std::string filename);

		//void setElastoplasticitySolver(std::shared_ptr<ElastoplasticityModule<TDataType>> solver);

		std::shared_ptr<Node> getSurfaceNode() { return m_surfaceNode; }

	protected:
		std::shared_ptr<Node> m_surfaceNode;

		//std::shared_ptr<ElasticityModule<TDataType>> m_elasticity;
		std::shared_ptr<HyperelastoplasticityModule<TDataType>> m_plasticity;
		std::shared_ptr<HyperelasticFractureModule<TDataType>> m_fracture;
		std::shared_ptr<DensityPBD<TDataType>> m_pbdModule;
		std::shared_ptr<ImplicitViscosity<TDataType>> m_visModule;
	};


#ifdef PRECISION_FLOAT
	template class HyperelastoplasticityBody<DataType3f>;
#else
	template class HyperelastoplasticityBody<DataType3d>;
#endif
}