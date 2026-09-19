"""
asymmetric_recurrent.py
=======================

The V4 policy: a recurrent actor-critic where the actor is partially
observed, the critic is privileged, and a state-estimation head hangs off the
actor's recurrent trunk.

This is the network in Fig. 3 and Fig. 4 of Shin et al. (RA-L 2026), adapted
to an ArUco front-end instead of their keypoint encoder.

    obs['actor']  13-D  --> LSTM_pi --> h_t --+--> MLP_pi --> action
                                              |
                                              +--> est head --> s_rel_hat (6)
                                                   (with the raw actor input
                                                    concatenated, as in Fig. 4)

    obs['critic'] 15-D  --> LSTM_vf --> MLP_vf --> value      [PRIVILEGED]


THE ONE THING THAT MUST NOT BE WRONG
------------------------------------------------------------------------------
The actor must never see obs['critic'], which contains the true relative
state. If it does, V4 is not a vision-based landing policy at all and every
number it produces is meaningless -- and it would still train, and still look
plausible, which is what makes this the dangerous failure.

There is therefore exactly ONE place in this file where the two streams are
separated, `_split_features`, and it is a pair of slices on a fixed layout.
Both the layout constants and the slice live here and nowhere else. If you
change the observation, change them here and check this file only.

`obs['target']` is the auxiliary-loss LABEL. It is in the observation space so
that it lands in the rollout buffer; it is deliberately NOT part of the
features tensor, so no code path can route it into either network. The trainer
reads it from `rollout_buffer.observations['target']`.


WHY THE EXTRACTOR CONCATENATES AND THEN WE SLICE
------------------------------------------------------------------------------
The obvious design is two separate features extractors (SB3 supports this via
`share_features_extractor=False`). It was not used because SB3 builds both
extractors from one class and one kwargs dict, so making them differ requires
either reaching into `_build` or replacing attributes after construction --
both of which put the privilege boundary somewhere implicit.

Concatenating in a known order and slicing once is more code but the boundary
is a single visible expression. Given what a leak would cost, that trade is
worth it.


WHAT IS REPLACED AFTER super().__init__()
------------------------------------------------------------------------------
SB3 sizes both LSTMs from `features_dim`, which here is the CONCATENATED width
(28). The actor LSTM must take 13 and the critic LSTM 15, so both are rebuilt
after construction, along with the estimation head.

Rebuilding modules invalidates the optimizer that SB3 already created over the
old parameters, so the optimizer is rebuilt too. Forgetting that step trains
nothing and raises no error -- the loss simply never moves.

`mlp_extractor` is NOT rebuilt: SB3 sizes it from `lstm_output_dim` (the hidden
size), not from `features_dim`, so it is already correct.
"""

import torch as th
from torch import nn

from sb3_contrib.common.recurrent.policies import (
    RecurrentMultiInputActorCriticPolicy)
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from envs.ShinLanderAviary import ACTOR_DIM, CRITIC_DIM, TARGET_DIM


class ActorCriticSplitExtractor(BaseFeaturesExtractor):
    """Concatenate ['actor', 'critic'] in that fixed order.

    'target' is dropped here on purpose. It is a label, and the only way to
    guarantee no network reads it is for it never to enter the features
    tensor.
    """

    def __init__(self, observation_space):
        super().__init__(observation_space,
                         features_dim=ACTOR_DIM + CRITIC_DIM)
        a = observation_space['actor'].shape[0]
        c = observation_space['critic'].shape[0]
        t = observation_space['target'].shape[0]
        if (a, c, t) != (ACTOR_DIM, CRITIC_DIM, TARGET_DIM):
            raise ValueError(
                f'observation space disagrees with ShinLanderAviary: got '
                f'actor={a}, critic={c}, target={t}; expected '
                f'{ACTOR_DIM}, {CRITIC_DIM}, {TARGET_DIM}. The slice in '
                f'AsymmetricRecurrentPolicy._split_features would be wrong.')

    def forward(self, observations):
        return th.cat([observations['actor'], observations['critic']], dim=1)


class AsymmetricRecurrentPolicy(RecurrentMultiInputActorCriticPolicy):
    """Recurrent actor-critic with a privileged critic and an estimation head.

    Constructor arguments beyond SB3's:
        lstm_hidden_size  hidden width of BOTH LSTMs (default 128, matching
                          the V3b offline estimator rather than the paper's
                          512 -- the input here is 13 metric numbers, not a
                          512-dim image embedding)
        n_lstm_layers     default 1, as V3b
        est_hidden        width of the estimation head's hidden layer
                          (default 256, the paper's y_t dimensionality)
    """

    def __init__(self, observation_space, action_space, lr_schedule,
                 lstm_hidden_size: int = 128,
                 n_lstm_layers: int = 1,
                 est_hidden: int = 256,
                 **kwargs):

        # Forced, not defaulted. Each of these is load-bearing:
        #   enable_critic_lstm=True   the critic needs its own recurrence;
        #                             sharing the actor's would give the
        #                             value function the actor's partial
        #                             observation and defeat the point
        #   shared_lstm=False         same reason
        #   share_features_extractor  one extractor produces the concatenated
        #     =True                   tensor that _split_features slices
        kwargs['enable_critic_lstm'] = True
        kwargs['shared_lstm'] = False
        kwargs['share_features_extractor'] = True
        kwargs['features_extractor_class'] = ActorCriticSplitExtractor
        kwargs['features_extractor_kwargs'] = {}

        super().__init__(
            observation_space, action_space, lr_schedule,
            lstm_hidden_size=lstm_hidden_size,
            n_lstm_layers=n_lstm_layers,
            **kwargs)

        # ---- rebuild the LSTMs at the correct input widths -------------
        # SB3 sized both from features_dim (28). See the module docstring.
        lstm_kwargs = dict(kwargs.get('lstm_kwargs') or {})
        self.lstm_actor = nn.LSTM(ACTOR_DIM, lstm_hidden_size,
                                  num_layers=n_lstm_layers, **lstm_kwargs)
        self.lstm_critic = nn.LSTM(CRITIC_DIM, lstm_hidden_size,
                                   num_layers=n_lstm_layers, **lstm_kwargs)

        # ---- estimation head -------------------------------------------
        # Fig. 4: the hidden state is concatenated with the image embedding
        # and the proprioceptive state before the estimation head. Here the
        # raw actor features play the role of [l_t, u_t].
        self.est_net = nn.Sequential(
            nn.Linear(lstm_hidden_size + ACTOR_DIM, est_hidden),
            nn.ReLU(),
            nn.Linear(est_hidden, TARGET_DIM),
        )

        self.to(self.device)

        # ---- rebuild the optimizer over the NEW parameters -------------
        # Without this the optimizer holds references to the discarded
        # LSTMs and never sees est_net. Training would run, report a loss,
        # and change nothing.
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        # Populated by forward() so the rollout collector can record the
        # per-step estimation error without a second forward pass. Shape
        # (n_envs, TARGET_DIM), detached.
        self._last_estimate = None

    # ==================================================================
    # THE PRIVILEGE BOUNDARY
    # ==================================================================

    @staticmethod
    def _split_features(features):
        """The only place actor and critic streams are separated.

        `features` is [actor | critic] as built by ActorCriticSplitExtractor.
        """
        pi = features[..., :ACTOR_DIM]
        vf = features[..., ACTOR_DIM:ACTOR_DIM + CRITIC_DIM]
        return pi, vf

    # ==================================================================
    # SHARED FORWARD BODY
    # ==================================================================

    def _latents(self, obs, lstm_states, episode_starts):
        """Run both recurrent trunks and the estimation head once."""
        features = self.extract_features(obs)
        pi_features, vf_features = self._split_features(features)

        latent_pi_lstm, states_pi = self._process_sequence(
            pi_features, lstm_states.pi, episode_starts, self.lstm_actor)
        latent_vf_lstm, states_vf = self._process_sequence(
            vf_features, lstm_states.vf, episode_starts, self.lstm_critic)

        # Estimation head reads the actor trunk ONLY. It is supervised by
        # ground truth, but it is not fed any.
        est = self.est_net(
            th.cat([latent_pi_lstm, pi_features], dim=1))

        latent_pi = self.mlp_extractor.forward_actor(latent_pi_lstm)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf_lstm)
        return latent_pi, latent_vf, est, RNNStates(states_pi, states_vf)

    # ==================================================================
    # SB3 INTERFACE
    # ==================================================================

    def forward(self, obs, lstm_states, episode_starts, deterministic=False):
        latent_pi, latent_vf, est, states = self._latents(
            obs, lstm_states, episode_starts)

        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        # Cached rather than returned: SB3's collect_rollouts expects
        # exactly four return values here, and changing that signature
        # would mean reimplementing the collector.
        self._last_estimate = est.detach()

        return actions, values, log_prob, states

    def evaluate_actions_with_estimate(self, obs, actions, lstm_states,
                                       episode_starts):
        """As evaluate_actions, plus the state estimate for the aux loss."""
        latent_pi, latent_vf, est, _ = self._latents(
            obs, lstm_states, episode_starts)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy(), est

    def evaluate_actions(self, obs, actions, lstm_states, episode_starts):
        values, log_prob, entropy, _ = self.evaluate_actions_with_estimate(
            obs, actions, lstm_states, episode_starts)
        return values, log_prob, entropy

    def get_distribution(self, obs, lstm_states, episode_starts):
        """Actor path only. `lstm_states` is the pi tuple, not RNNStates."""
        features = self.extract_features(obs)
        pi_features, _ = self._split_features(features)
        latent_pi, states = self._process_sequence(
            pi_features, lstm_states, episode_starts, self.lstm_actor)
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        return self._get_action_dist_from_latent(latent_pi), states

    def predict_values(self, obs, lstm_states, episode_starts):
        """Critic path only. `lstm_states` is the vf tuple, not RNNStates."""
        features = self.extract_features(obs)
        _, vf_features = self._split_features(features)
        latent_vf, _ = self._process_sequence(
            vf_features, lstm_states, episode_starts, self.lstm_critic)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        return self.value_net(latent_vf)

    # ==================================================================
    # DEPLOYMENT CHECK
    # ==================================================================

    def actor_parameter_names(self):
        """Names of the parameters that would ship on the drone.

        The privileged branch (lstm_critic, mlp_extractor.value_net path,
        value_net) is removed at deployment, as in the paper. This exists so
        a test can assert that nothing in the actor path depends on it.
        """
        keep = ('lstm_actor', 'mlp_extractor.policy_net', 'action_net',
                'est_net', 'log_std')
        return [n for n, _ in self.named_parameters()
                if any(n.startswith(k) for k in keep)]
