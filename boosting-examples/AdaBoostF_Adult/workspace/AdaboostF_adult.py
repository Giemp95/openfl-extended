import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier

from AdultDataset import AdultDataset
from adaboost import AdaBoostF
from openfl.interface.interactive_api.experiment import FLExperiment, TaskInterface, ModelInterface
from openfl.interface.interactive_api.federation import Federation

random_state = np.random.RandomState(31415)

client_id = 'api'
cert_dir = '../cert'
director_node_fqdn = 'localhost'

task_interface = TaskInterface()


@task_interface.register_fl_task(model='model', data_loader='train_loader', device='device', optimizer='optimizer',
                                 adaboost_coeff='adaboost_coeff')
def train_adaboost(model, train_loader, device, optimizer, adaboost_coeff):
    X, y = train_loader
    X, y = np.array(X), np.array(y)
    adaboost_coeff = np.array(adaboost_coeff)

    weak_learner = model.get(0)
    ids = np.random.choice(X.shape[0], size=X.shape[0], replace=True, p=adaboost_coeff / adaboost_coeff.sum())
    weak_learner.fit(X[ids], y[ids])

    metric = accuracy_score(y, weak_learner.predict(X))

    return {'accuracy': metric}


@task_interface.register_fl_task(model='model', data_loader='val_loader', device='device',
                                 adaboost_coeff='adaboost_coeff')
def validate_weak_learners(model, val_loader, device, adaboost_coeff):
    X, y = val_loader
    X, y = np.array(X), np.array(y)
    adaboost_coeff = np.array(adaboost_coeff)

    error = []
    miss = []
    for weak_learner in model.get_estimators():
        pred = weak_learner.predict(X)
        mispredictions = y != pred
        error.append(sum(adaboost_coeff[mispredictions]))
        miss.append(mispredictions)
    # TODO: piccolo trick, alla fine di ogni vettore errori viene mandata la norma dei pesi locali
    error.append(sum(adaboost_coeff))

    return {'errors': error}, {'misprediction': miss}


@task_interface.register_fl_task(model='model', data_loader='val_loader', device='device')
def adaboost_update(model, val_loader, device):
    return {'adaboost_update': 0}


@task_interface.register_fl_task(model='model', data_loader='val_loader', device='device')
def validate(model, val_loader, device):
    X, y = val_loader
    print(model.get_estimator_number())
    pred = model.predict(np.array(X))
    f1 = f1_score(y, pred, average="macro")

    return {'F1 score': f1}


federation = Federation(client_id=client_id, director_node_fqdn=director_node_fqdn, director_port='50052', tls=False)
fl_experiment = FLExperiment(federation=federation, experiment_name="AdaboostF_Adult",
                             serializer_plugin='openfl.plugins.interface_serializer.dill_serializer.DillSerializer',
                             load_default_plan=False)
model_interface = ModelInterface(
    model=AdaBoostF(base_estimator=DecisionTreeClassifier(max_leaf_nodes=10)),
    optimizer=None,
    framework_plugin='openfl.plugins.frameworks_adapters.generic_adapter.GenericAdapter')
federated_dataset = AdultDataset()

fl_experiment.start(
    model_provider=model_interface,
    task_keeper=task_interface,
    data_loader=federated_dataset,
    rounds_to_train=30,
    opt_treatment='CONTINUE_GLOBAL',
    nn=False,
)

fl_experiment.stream_metrics(tensorboard_logs=False)
fl_experiment.remove_experiment_data()
