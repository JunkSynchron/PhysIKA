#include <iostream>
#include <memory>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "GUI/GlutGUI/GLApp.h"

#include "Framework/Framework/SceneGraph.h"
#include "Framework/Topology/PointSet.h"
#include "Framework/Framework/Log.h"

#include "Dynamics/ParticleSystem/ParticleElasticBody.h"
#include "Dynamics/ParticleSystem/StaticBoundary.h"
#include "Dynamics/ParticleSystem/HyperelasticityModule.h"
#include "Dynamics/ParticleSystem/HyperelasticityModule_test.h"
#include "Dynamics/ParticleSystem/HyperelasticityModule_NewtonMethod.h"
#include "Rendering/SurfaceMeshRender.h"

#include "Dynamics/ParticleSystem/ParticleIntegrator.h"
#include "BarStretch.h"

using namespace std;
using namespace Physika;

void RecieveLogMessage(const Log::Message& m)
{
	switch (m.type)
	{
	case Log::Info:
		cout << ">>>: " << m.text << endl; break;
	case Log::Warning:
		cout << "???: " << m.text << endl; break;
	case Log::Error:
		cout << "!!!: " << m.text << endl; break;
	case Log::User:
		cout << ">>>: " << m.text << endl; break;
	default: break;
	}
}

void CreateScene()
{
	SceneGraph& scene = SceneGraph::getInstance();

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
	root->loadCube(Vector3f(0), Vector3f(1.0), 0.005, true);

	std::shared_ptr<ParticleElasticBody<DataType3f>> child3 = std::make_shared<ParticleElasticBody<DataType3f>>();
	root->addParticleSystem(child3);
	child3->getRenderModule()->setColor(Vector3f(0, 1, 1));
	child3->setMass(1.0);

	Real p_distance = 0.005;
	Real half_p_distance = 0.5 * p_distance;

	Vector3f bar_particles(0.03, 0.01, 0.01);
	child3->loadParticles( -bar_particles + half_p_distance, bar_particles, p_distance);

	Vector<int,3> bar_size;
	bar_size[0] = (2 * bar_particles[0]) / p_distance;
	bar_size[1] = (2 * bar_particles[1]) / p_distance;
	bar_size[2] = (2 * bar_particles[2]) / p_distance;

	//child3->loadParticles("../Media/bunny/bunny_points.obj");
	//child3->loadSurface("../Media/bunny/bunny_mesh.obj");
	child3->setDt(0.00005f);
	child3->translate(Vector3f(0.5, 0.2, 0.5));
	child3->setVisible(true);
	auto hyper_test = std::make_shared<HyperelasticityModule_test<DataType3f>>();
	hyper_test->setMu(300);
	hyper_test->setLambda(100);
	hyper_test->setMethodImplicit();
	child3->setElasticitySolver(hyper_test);

	/*{
		std::shared_ptr<HyperelasticityModule_test<DataType3f>> ptr_HM_module =
			child3->template getModule<HyperelasticityModule_test<DataType3f>>("elasticity");
		if (ptr_HM_module != nullptr) {
			printf("Set Initial Stretch!\n");
			ptr_HM_module->setInitialStretch(0.1);
		}
	}*/

	child3->getSurfaceRender()->setColor(Vector3f(1, 1, 0));
	child3->getElasticitySolver()->setIterationNumber(10);

	/*
	std::shared_ptr<ParticleElasticBody<DataType3f>> child4 = std::make_shared<ParticleElasticBody<DataType3f>>();
	root->addParticleSystem(child4);
	child4->getRenderModule()->setColor(Vector3f(0, 1, 1));
	child4->setMass(1.0);
	child4->loadParticles("../Media/bunny/bunny_points.obj");
	child4->loadSurface("../Media/bunny/bunny_mesh.obj");
	child4->scale(0.7);
	child4->translate(Vector3f(0.5, 0.2, 0.5));
	child4->setVisible(false);
	auto hyper = std::make_shared<HyperelasticityModule<DataType3f>>();
	hyper->setEnergyFunction(HyperelasticityModule<DataType3f>::Quadratic);
	child4->setElasticitySolver(hyper);
	child4->getSurfaceRender()->setColor(Vector3f(0, 1, 0.5));
	child4->getElasticitySolver()->setIterationNumber(10);
	*/

	/*
	std::shared_ptr<ParticleElasticBody<DataType3f>> child5 = std::make_shared<ParticleElasticBody<DataType3f>>();
	root->addParticleSystem(child5);
	child5->getRenderModule()->setColor(Vector3f(0, 1, 1));
	child5->setMass(1.0);
	child5->loadParticles("../Media/bunny/bunny_points.obj");
	child5->loadSurface("../Media/bunny/bunny_mesh.obj");
	child5->scale(0.7);
	child5->translate(Vector3f(0.75, 0.2, 0.5));
	child5->setVisible(false);
	child5->getSurfaceRender()->setColor(Vector3f(1,0,0));
	*/
}

void CreateScene_NewtonMethod() {
	SceneGraph& scene = SceneGraph::getInstance();

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
	root->loadCube(Vector3f(0), Vector3f(1.0), 0.005, true);

	std::shared_ptr<ParticleElasticBody<DataType3f>> child3 = std::make_shared<ParticleElasticBody<DataType3f>>();
	root->addParticleSystem(child3);
	child3->getRenderModule()->setColor(Vector3f(0, 1, 1));
	child3->setMass(1.0);

	Real p_distance = 0.01;
	Real half_p_distance = 0.5 * p_distance;

	Vector3f bar_particles(0.03, 0.05, 0.03);
	child3->loadParticles(-bar_particles + half_p_distance, bar_particles, p_distance);

	Vector<int, 3> bar_size;
	bar_size[0] = (2 * bar_particles[0]) / p_distance;
	bar_size[1] = (2 * bar_particles[1]) / p_distance;
	bar_size[2] = (2 * bar_particles[2]) / p_distance;

	//child3->loadParticles("../Media/bunny/bunny_points.obj");
	//child3->loadSurface("../Media/bunny/bunny_mesh.obj");
	child3->setDt(0.0001f);
	child3->translate(Vector3f(0.5, 0.5, 0.5));
	child3->setVisible(true);
	auto hyper_test = std::make_shared<HyperelasticityModule_NewtonMethod<DataType3f>>();
	hyper_test->setMu(50);
	hyper_test->setLambda(20);
	child3->setElasticitySolver(hyper_test);

	auto bar_stretch = std::make_shared<BarStretchIntegrator<DataType3f>>();
	bar_stretch->setName("integrator");
	bar_stretch->setBarSize(bar_size);
	bar_stretch->setInitialStretch(1.1);
	bar_stretch->disableGravity();
	bar_stretch->setRelative_YPlane(0.5);
	child3->setBarIntegrator(bar_stretch);
	/*{
	std::shared_ptr<HyperelasticityModule_test<DataType3f>> ptr_HM_module =
	child3->template getModule<HyperelasticityModule_test<DataType3f>>("elasticity");
	if (ptr_HM_module != nullptr) {
	printf("Set Initial Stretch!\n");
	ptr_HM_module->setInitialStretch(0.1);
	}
	}*/
	child3->getSurfaceRender()->setColor(Vector3f(1, 1, 0));

}



int main()
{
	CreateScene_NewtonMethod();

	Log::setOutput("console_log.txt");
	Log::setLevel(Log::Info);
	Log::setUserReceiver(&RecieveLogMessage);
	Log::sendMessage(Log::Info, "Simulation begin");

	GLApp window;
	window.createWindow(1024, 768);

	window.mainLoop();

	Log::sendMessage(Log::Info, "Simulation end!");
	return 0;
}


