/**
 * @file Node.h
 * @author Xiaowei He (xiaowei@iscas.ac.cn)
 * @brief A tree node, it may contain a number of different kinds of fields, modules and child nodes
 * @version 0.1
 * @date 2019-06-08
 * 
 * @copyright Copyright (c) 2019
 * 
 */
#pragma once
#include "Base.h"
#include "Core/Typedef.h"
#include "FieldVar.h"
#include "Core/Platform.h"
#include "NumericalModel.h"
#include "ModuleTopology.h"
#include "DeviceContext.h"
#include "MechanicalState.h"
#include "ModuleForce.h"
#include "ModuleConstraint.h"
#include "CollisionModel.h"
#include "CollidableObject.h"
#include "ModuleVisual.h"
#include "ControllerAnimation.h"
#include "ControllerRender.h"
#include "TopologyMapping.h"
#include "NumericalIntegrator.h"
#include "ModuleCompute.h"

namespace Physika {
class Action;

class Node : public Base
{
	DECLARE_CLASS(Node)
public:

	template<class T>
	using SPtr = std::shared_ptr<T>;

	Node(std::string name = "default");
	virtual ~Node();

	void setName(std::string name);
	std::string getName();

	Node* getChild(std::string name);
	Node* getParent();
	Node* getRoot();

	/// Check the state of dynamics
	virtual bool isActive();

	/// Set the state of dynamics
	virtual void setActive(bool active);

	/// Check the visibility of context
	virtual bool isVisible();

	/// Set the visibility of context
	virtual void setVisible(bool visible);

	/// Simulation time
	virtual Real getTime();

	/// Simulation timestep
	virtual Real getDt();

	void setDt(Real dt);

	void setMass(Real mass);
	Real getMass();

	/**
	 * @brief Create a Child object
	 * 
	 * @tparam TNode 						Node type of the child object
	 * @param name 							Node name
	 * @return std::shared_ptr<TNode> 		return the created child, if name is aleady used, return nullptr.
	 */
	template<class TNode>
	std::shared_ptr<TNode> createChild(std::string name)
	{
		return addChild(TypeInfo::New<TNode>(name));
	}

	/**
	 * @brief Add a child
	 * 
	 * @param child 
	 * @return std::shared_ptr<Node> 
	 */
	std::shared_ptr<Node> addChild(std::shared_ptr<Node> child) {
		m_children.push_back(child);
		child->setParent(this);
		return child;
	}

	void removeChild(std::shared_ptr<Node> child);

	/**
	 * @brief Return all children
	 * 
	 * @return ListPtr<Node> Children list
	 */
	ListPtr<Node> getChildren() { return m_children; }


	std::shared_ptr<DeviceContext> getContext();
	void setContext(std::shared_ptr<DeviceContext> context);
	virtual void setAsCurrentContext();

	std::shared_ptr<MechanicalState> getMechanicalState();
	void setMechanicalState(std::shared_ptr<MechanicalState> state);

	/**
	 * @brief Add a module to m_module_list and other special module lists
	 * 
	 * @param module 	module should be created before calling this function
	 * @return true 	return true if successfully added
	 * @return false 	return false if module already exists
	 */
	bool addModule(std::shared_ptr<Module> module);
	bool deleteModule(std::shared_ptr<Module> module);

	/**
	 * @brief Add a speical kind of module
	 * 
	 * @tparam TModule 	Module type
	 * @param tModule 	Added module
	 * @return true 
	 * @return false 
	 */
	template<class TModule>
	bool addModule(std::shared_ptr<TModule> tModule)
	{
		std::shared_ptr<Module> module = std::dynamic_pointer_cast<Module>(tModule);
		return addModule(module);
	}
	template<class TModule>
	bool deleteModule(std::shared_ptr<TModule> tModule)
	{
		std::shared_ptr<Module> module = std::dynamic_pointer_cast<Module>(tModule);
		return deleteModule(module);
	}

	std::list<std::shared_ptr<Module>>& getModuleList() { return m_module_list; }

	bool hasModule(std::string name);

	/**
	 * @brief Get a module by its name
	 * 
	 * @param name 	Module name
	 * @return std::shared_ptr<Module>	return nullptr is no module is found, otherwise return the first found module
	 */
	std::shared_ptr<Module> getModule(std::string name);

	/**
	 * @brief Get the Module by the module class name
	 * 
	 * @tparam TModule 	Module class name
	 * @return std::shared_ptr<TModule> return nullptr is no module is found, otherwise return the first found module
	 */
	template<class TModule>
	std::shared_ptr<TModule> getModule()
	{
		TModule* tmp = new TModule;
		std::shared_ptr<Module> base;
		std::list<std::shared_ptr<Module>>::iterator iter;
		for (iter = m_module_list.begin(); iter != m_module_list.end(); iter++)
		{
			if ((*iter)->getClassInfo() == tmp->getClassInfo())
			{
				base = *iter;
				break;
			}
		}
		delete tmp;
		return TypeInfo::CastPointerDown<TModule>(base);
	}

	template<class TModule> 
	std::shared_ptr<TModule> getModule(std::string name)
	{
		std::shared_ptr<Module> base = getModule(name);

		return TypeInfo::CastPointerDown<TModule>(base);
	}


#define NODE_SET_SPECIAL_MODULE( CLASSNAME )												\
	template<class TModule>																	\
	std::shared_ptr<TModule> set##CLASSNAME(std::string name) {								\
		if (hasModule(name))																					\
		{																										\
			Log::sendMessage(Log::Error, std::string("Module ") + name + std::string(" already exists!"));		\
			return nullptr;																						\
		}																										\
																												\
		std::shared_ptr<TModule> module = std::make_shared<TModule>();											\
		module->setName(name);																					\
		this->set##CLASSNAME(module);																			\
																												\
		return module;																							\
	}
	NODE_SET_SPECIAL_MODULE(TopologyModule)
	NODE_SET_SPECIAL_MODULE(NumericalModel)
	NODE_SET_SPECIAL_MODULE(CollidableObject)
	NODE_SET_SPECIAL_MODULE(NumericalIntegrator)

#define NODE_SET_SPECIAL_MODULE( CLASSNAME, MODULENAME )						\
	virtual void set##CLASSNAME( std::shared_ptr<CLASSNAME> module) {			\
		if(MODULENAME != nullptr)	deleteFromModuleList(MODULENAME);					\
		MODULENAME = module;													\
		addToModuleList(module);														\
	}

	NODE_SET_SPECIAL_MODULE(TopologyModule, m_topology)
	NODE_SET_SPECIAL_MODULE(NumericalModel, m_numerical_model)
	NODE_SET_SPECIAL_MODULE(CollidableObject, m_collidable_object)
	NODE_SET_SPECIAL_MODULE(NumericalIntegrator, m_numerical_integrator)


	std::shared_ptr<CollidableObject>		getCollidableObject() { return m_collidable_object; }
	std::shared_ptr<NumericalModel>			getNumericalModel() { return m_numerical_model; }
	std::shared_ptr<TopologyModule>			getTopologyModule() { return m_topology; }
	std::shared_ptr<NumericalIntegrator>	getNumericalIntegrator() { return m_numerical_integrator; }


	template<class TModule>
	std::shared_ptr<TModule> addModule(std::string name)
	{
		if (hasModule(name))
		{
			Log::sendMessage(Log::Error, std::string("Module ") + name + std::string(" already exists!"));
			return nullptr;
		}
		std::shared_ptr<TModule> module = std::make_shared<TModule>();
		module->setName(name);
		this->addModule(module);

		return module;
	}

#define NODE_CREATE_SPECIAL_MODULE(CLASSNAME) \
	template<class TModule>									\
		std::shared_ptr<TModule> add##CLASSNAME(std::string name)	\
		{																	\
			if (hasModule(name))											\
			{																\
				Log::sendMessage(Log::Error, std::string("Module ") + name + std::string(" already exists!"));	\
				return nullptr;																					\
			}																									\
			std::shared_ptr<TModule> module = std::make_shared<TModule>();										\
			module->setName(name);																				\
			this->add##CLASSNAME(module);																		\
																												\
			return module;																						\
		}																										

#define NODE_ADD_SPECIAL_MODULE( CLASSNAME, SEQUENCENAME ) \
	virtual void add##CLASSNAME( std::shared_ptr<CLASSNAME> module) { SEQUENCENAME.push_back(module); addToModuleList(module);} \
	virtual void delete##CLASSNAME( std::shared_ptr<CLASSNAME> module) { SEQUENCENAME.remove(module); deleteFromModuleList(module); } \
	std::list<std::shared_ptr<CLASSNAME>>& get##CLASSNAME##List(){ return SEQUENCENAME;}

	NODE_ADD_SPECIAL_MODULE(ForceModule, m_force_list)
		NODE_ADD_SPECIAL_MODULE(ConstraintModule, m_constraint_list)
		NODE_ADD_SPECIAL_MODULE(CollisionModel, m_collision_list)
		NODE_ADD_SPECIAL_MODULE(VisualModule, m_render_list)
		NODE_ADD_SPECIAL_MODULE(TopologyMapping, m_topology_mapping_list)
		NODE_ADD_SPECIAL_MODULE(ComputeModule, m_compute_list)

		NODE_CREATE_SPECIAL_MODULE(ForceModule)
		NODE_CREATE_SPECIAL_MODULE(ConstraintModule)
		NODE_CREATE_SPECIAL_MODULE(CollisionModel)
		NODE_CREATE_SPECIAL_MODULE(VisualModule)
		NODE_CREATE_SPECIAL_MODULE(TopologyMapping)
		NODE_CREATE_SPECIAL_MODULE(ComputeModule)

	virtual bool initialize() { return true; }
	virtual void draw() {};
	virtual void advance(Real dt);
	virtual void takeOneFrame() {};
	virtual void updateModules() {};
	virtual void updateTopology() {};
	virtual void updateTopologyMapping() {};
	virtual bool resetStatus() { return true; }

	/**
	 * @brief Depth-first tree traversal 
	 * 
	 * @param act 	Operation on the node
	 */
	void traverseBottomUp(Action* act);
	template<class Act>
	void traverseBottomUp() {
		Act action;
		doTraverseBottomUp(&action);
	}

	/**
	 * @brief Breadth-first tree traversal
	 * 
	 * @param act 	Operation on the node
	 */
	void traverseTopDown(Action* act);
	template<class Act>
	void traverseTopDown() {
		Act action;
		doTraverseTopDown(&action);
	}

protected:
	void setParent(Node* p) { m_parent = p; }

	virtual void doTraverseBottomUp(Action* act);
	virtual void doTraverseTopDown(Action* act);

private:
	bool addToModuleList(std::shared_ptr<Module> module);
	bool deleteFromModuleList(std::shared_ptr<Module> module);

#define NODE_ADD_SPECIAL_MODULE_LIST( CLASSNAME, SEQUENCENAME ) \
	virtual void addTo##CLASSNAME##List( std::shared_ptr<CLASSNAME> module) { SEQUENCENAME.push_back(module); } \
	virtual void deleteFrom##CLASSNAME##List( std::shared_ptr<CLASSNAME> module) { SEQUENCENAME.remove(module); } \

	NODE_ADD_SPECIAL_MODULE_LIST(ForceModule, m_force_list)
		NODE_ADD_SPECIAL_MODULE_LIST(ConstraintModule, m_constraint_list)
		NODE_ADD_SPECIAL_MODULE_LIST(CollisionModel, m_collision_list)
		NODE_ADD_SPECIAL_MODULE_LIST(VisualModule, m_render_list)
		NODE_ADD_SPECIAL_MODULE_LIST(TopologyMapping, m_topology_mapping_list)
		NODE_ADD_SPECIAL_MODULE_LIST(ComputeModule, m_compute_list)

private:
	/**
	 * @brief Time step size
	 * 
	 */
	Real m_dt;
	bool m_initalized;

	VarField<Real> m_mass;
	/**
	 * @brief Dynamics indicator
	 * true: Dynamics is turn on
	 * false: Dynamics is turned off
	 */
	VarField<bool> m_active;
	/**
	 * @brief Visibility
	 * true: the node is visible
	 * false: the node is invisible
	 */
	VarField<bool> m_visible;
	VarField<Real> m_time;

	VarField<std::string> m_node_name;

	/**
	 * @brief A module list containing all modules
	 * 
	 */
	std::list<std::shared_ptr<Module>> m_module_list;

	/**
	 * @brief Pointer of a specific module
	 * 
	 */
	std::shared_ptr<TopologyModule> m_topology;
	std::shared_ptr<NumericalModel> m_numerical_model;
	std::shared_ptr<MechanicalState> m_mechanical_state;
	std::shared_ptr<CollidableObject> m_collidable_object;
	std::shared_ptr<NumericalIntegrator> m_numerical_integrator;

	/**
	 * @brief A module list containg specific modules
	 * 
	 */
	std::list<std::shared_ptr<ForceModule>> m_force_list;
	std::list<std::shared_ptr<ConstraintModule>> m_constraint_list;
	std::list<std::shared_ptr<ComputeModule>> m_compute_list;
	std::list<std::shared_ptr<CollisionModel>> m_collision_list;
	std::list<std::shared_ptr<VisualModule>> m_render_list;
	std::list<std::shared_ptr<TopologyMapping>> m_topology_mapping_list;

	std::shared_ptr<DeviceContext> m_context;

	ListPtr<Node> m_children;

	/**
	 * @brief Indicating which node the current module belongs to
	 * 
	 */
	Node* m_parent;
};
}
