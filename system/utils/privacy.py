from opacus import PrivacyEngine
from opacus.dp_model_inspector import DPModelInspector

MAX_GRAD_NORM = 1.0
EPSILON = 50.0
DELTA = 1e-7
EPOCHS = 1 * 100
N_ACCUMULATION_STEPS = 4

def initialize_dp(model, optimizer, sample_rate, dp_sigma):
    privacy_engine = PrivacyEngine(
        model,
        sample_rate = sample_rate * N_ACCUMULATION_STEPS,
        # epochs = EPOCHS,
        # target_epsilon = EPSILON,
        target_delta = DELTA,
        noise_multiplier = dp_sigma, 
        max_grad_norm = MAX_GRAD_NORM,
    )
    privacy_engine.attach(optimizer)


def get_dp_params(optimizer):
    return optimizer.privacy_engine.get_privacy_spent(DELTA), DELTA


def check_dp(model):
    inspector = DPModelInspector()
    inspector.validate(model)


def dp_step(optimizer, i, len_train_loader):
    # take a real optimizer step after N_VIRTUAL_STEP steps t
    if ((i + 1) % N_ACCUMULATION_STEPS == 0) or ((i + 1) == len_train_loader):
        optimizer.step()
    else:
        optimizer.virtual_step() # take a virtual step