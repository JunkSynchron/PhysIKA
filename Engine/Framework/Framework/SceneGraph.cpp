#include "SceneGraph.h"
#include "Framework/Action/ActAnimate.h"
#include "Framework/Action/ActDraw.h"
#include "Framework/Action/ActInit.h"
#include "Framework/Framework/SceneLoaderFactory.h"

namespace Physika
{
SceneGraph& SceneGraph::getInstance()
{
	static SceneGraph m_instance;
	return m_instance;
}

void SceneGraph::setGravity(Vector3f g)
{
	m_gravity = g;
}

Vector3f SceneGraph::getGravity()
{
	return m_gravity;
}

bool SceneGraph::initialize()
{
	if (m_initialized)
	{
		return true;
	}
	//TODO: check initialization
	if (m_root == nullptr)
	{
		return false;
	}

	m_root->traverseBottomUp<InitAct>();
	m_initialized = true;

	return m_initialized;
}

void SceneGraph::draw()
{
	if (m_root == nullptr)
	{
		return;
	}

	m_root->traverseTopDown<DrawAct>();
}

void SceneGraph::advance(float dt)
{
//	AnimationController*  aController = m_root->getAnimationController();
	//	aController->
}

void SceneGraph::takeOneFrame()
{
	if (m_root == nullptr)
	{
		return;
	}

	m_root->traverseTopDown<AnimateAct>();
}

void SceneGraph::run()
{

}

bool SceneGraph::load(std::string name)
{
	SceneLoader* loader = SceneLoaderFactory::getInstance().getEntryByFileName(name);
	if (loader)
	{
		m_root = loader->load(name);
		return true;
	}

	return false;
}

Vector3f SceneGraph::getLowerBound()
{
	return m_lowerBound;
}

Vector3f SceneGraph::getUpperBound()
{
	return m_upperBound;
}

void SceneGraph::setLowerBound(Vector3f lowerBound)
{
	m_lowerBound = lowerBound;
}

void SceneGraph::setUpperBound(Vector3f upperBound)
{
	m_upperBound = upperBound;
}

}