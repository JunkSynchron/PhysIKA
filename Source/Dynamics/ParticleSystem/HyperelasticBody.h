#pragma once
#include "Framework/Framework/Node.h"
#include "TetSystem.h"

namespace PhysIKA
{
	template<typename> class ElasticityModule;
	template<typename> class PointSetToPointSet;
	template <typename> class UnstructuredPointSet;

	/*!
	*	\class	ParticleElasticBody
	*	\brief	Peridynamics-based elastic object.
	*/
	template<typename TDataType>
	class HyperelasticBody : public Node
	{
		DECLARE_CLASS_1(ParticleElasticBody, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		HyperelasticBody(std::string name = "default");
		virtual ~HyperelasticBody();

		bool initialize() override;
		void advance(Real dt) override;
		void updateTopology() override;
		bool resetStatus() override;

		bool translate(Coord t);
		bool scale(Real s);

		void setElasticitySolver(std::shared_ptr<ElasticityModule<TDataType>> solver);

		std::shared_ptr<ElasticityModule<TDataType>> getElasticitySolver();

		void loadFile(std::string filename);

		std::shared_ptr<TetSystem<TDataType>> getMeshNode() { return m_mesh_node; }

	public:
		DEF_EMPTY_VAR(Horizon, Real, "Horizon");

		/**
		 * @brief Particle position
		 */
		DEF_EMPTY_CURRENT_ARRAY(Position, Coord, DeviceType::GPU, "Particle position");


		/**
		 * @brief Particle velocity
		 */
		DEF_EMPTY_CURRENT_ARRAY(Velocity, Coord, DeviceType::GPU, "Particle velocity");

		/**
		 * @brief Particle force
		 */
		DEF_EMPTY_CURRENT_ARRAY(Force, Coord, DeviceType::GPU, "Force on each particle");

	private:
		std::shared_ptr<UnstructuredPointSet<TDataType>> m_pSet;

		std::shared_ptr<TetSystem<TDataType>> m_mesh_node;
	};


#ifdef PRECISION_FLOAT
	template class HyperelasticBody<DataType3f>;
#else
	template class HyperelasticBody<DataType3d>;
#endif
}