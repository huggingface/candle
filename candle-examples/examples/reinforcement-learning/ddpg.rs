use std::collections::VecDeque;

use candle::{DType, Device, Error, Module, Result, Tensor, Var};
use candle_nn::{
    func, linear, sequential::seq, Activation, AdamW, Optimizer, ParamsAdamW, Sequential,
    VarBuilder, VarMap,
};
use rand::{distr::Uniform, rng, Rng};

use super::gym_env::GymEnv;

pub struct OuNoise {
    mu: f64,
    theta: f64,
    sigma: f64,
    state: Tensor,
}
impl OuNoise {
    pub fn new(mu: f64, theta: f64, sigma: f64, size_action: usize) -> Result<Self> {
        Ok(Self {
            mu,
            theta,
            sigma,
            state: Tensor::ones(size_action, DType::F32, &Device::Cpu)?,
        })
    }

    pub fn sample(&mut self) -> Result<Tensor> {
        let rand = Tensor::randn_like(&self.state, 0.0, 1.0)?;
        let dx = ((self.theta * (self.mu - &self.state)?)? + (self.sigma * rand)?)?;
        self.state = (&self.state + dx)?;
        Ok(self.state.clone())
    }
}

#[derive(Clone)]
struct Transition {
    state: Tensor,
    action: Tensor,
    reward: Tensor,
    next_state: Tensor,
    terminated: bool,
    truncated: bool,
}
impl Transition {
    fn new(
        state: &Tensor,
        action: &Tensor,
        reward: &Tensor,
        next_state: &Tensor,
        terminated: bool,
        truncated: bool,
    ) -> Self {
        Self {
            state: state.clone(),
            action: action.clone(),
            reward: reward.clone(),
            next_state: next_state.clone(),
            terminated,
            truncated,
        }
    }
}

pub struct ReplayBuffer {
    buffer: VecDeque<Transition>,
    capacity: usize,
    size: usize,
}
impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            size: 0,
        }
    }

    pub fn push(
        &mut self,
        state: &Tensor,
        action: &Tensor,
        reward: &Tensor,
        next_state: &Tensor,
        terminated: bool,
        truncated: bool,
    ) {
        if self.size == self.capacity {
            self.buffer.pop_front();
        } else {
            self.size += 1;
        }
        self.buffer.push_back(Transition::new(
            state, action, reward, next_state, terminated, truncated,
        ));
    }

    #[allow(clippy::type_complexity)]
    pub fn random_batch(
        &self,
        batch_size: usize,
    ) -> Result<Option<(Tensor, Tensor, Tensor, Tensor, Vec<bool>, Vec<bool>)>> {
        if self.size < batch_size {
            Ok(None)
        } else {
            let transitions: Vec<&Transition> = rng()
                .sample_iter(Uniform::try_from(0..self.size).map_err(Error::wrap)?)
                .take(batch_size)
                .map(|i| self.buffer.get(i).unwrap())
                .collect();

            let states: Vec<Tensor> = transitions
                .iter()
                .map(|t| t.state.unsqueeze(0))
                .collect::<Result<_>>()?;
            let actions: Vec<Tensor> = transitions
                .iter()
                .map(|t| t.action.unsqueeze(0))
                .collect::<Result<_>>()?;
            let rewards: Vec<Tensor> = transitions
                .iter()
                .map(|t| t.reward.unsqueeze(0))
                .collect::<Result<_>>()?;
            let next_states: Vec<Tensor> = transitions
                .iter()
                .map(|t| t.next_state.unsqueeze(0))
                .collect::<Result<_>>()?;
            let terminateds: Vec<bool> = transitions.iter().map(|t| t.terminated).collect();
            let truncateds: Vec<bool> = transitions.iter().map(|t| t.truncated).collect();

            Ok(Some((
                Tensor::cat(&states, 0)?,
                Tensor::cat(&actions, 0)?,
                Tensor::cat(&rewards, 0)?,
                Tensor::cat(&next_states, 0)?,
                terminateds,
                truncateds,
            )))
        }
    }
}

fn track(
    varmap: &mut VarMap,
    vb: &VarBuilder,
    target_prefix: &str,
    network_prefix: &str,
    dims: &[(usize, usize)],
    tau: f64,
) -> Result<()> {
    for (i, &(in_dim, out_dim)) in dims.iter().enumerate() {
        let target_w = vb.get((out_dim, in_dim), &format!("{target_prefix}-fc{i}.weight"))?;
        let network_w = vb.get((out_dim, in_dim), &format!("{network_prefix}-fc{i}.weight"))?;
        varmap.set_one(
            format!("{target_prefix}-fc{i}.weight"),
            ((tau * network_w)? + ((1.0 - tau) * target_w)?)?,
        )?;

        let target_b = vb.get(out_dim, &format!("{target_prefix}-fc{i}.bias"))?;
        let network_b = vb.get(out_dim, &format!("{network_prefix}-fc{i}.bias"))?;
        varmap.set_one(
            format!("{target_prefix}-fc{i}.bias"),
            ((tau * network_b)? + ((1.0 - tau) * target_b)?)?,
        )?;
    }
    Ok(())
}

#[allow(unused)]
struct Actor<'a> {
    varmap: VarMap,
    vb: VarBuilder<'a>,
    network: Sequential,
    target_network: Sequential,
    size_state: usize,
    size_action: usize,
    dims: Vec<(usize, usize)>,
}

impl Actor<'_> {
    fn new(device: &Device, dtype: DType, size_state: usize, size_action: usize) -> Result<Self> {
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, device);

        let dims = vec![(size_state, 400), (400, 300), (300, size_action)];

        let make_network = |prefix: &str| {
            let seq = seq()
                .add(linear(
                    dims[0].0,
                    dims[0].1,
                    vb.pp(format!("{prefix}-fc0")),
                )?)
                .add(Activation::Relu)
                .add(linear(
                    dims[1].0,
                    dims[1].1,
                    vb.pp(format!("{prefix}-fc1")),
                )?)
                .add(Activation::Relu)
                .add(linear(
                    dims[2].0,
                    dims[2].1,
                    vb.pp(format!("{prefix}-fc2")),
                )?)
                .add(func(|xs| xs.tanh()));
            Ok::<Sequential, Error>(seq)
        };

        let network = make_network("actor")?;
        let target_network = make_network("target-actor")?;

        // this sets the two networks to be equal to each other using tau = 1.0
        track(&mut varmap, &vb, "target-actor", "actor", &dims, 1.0)?;

        Ok(Self {
            varmap,
            vb,
            network,
            target_network,
            size_state,
            size_action,
            dims,
        })
    }

    fn forward(&self, state: &Tensor) -> Result<Tensor> {
        self.network.forward(state)
    }

    fn target_forward(&self, state: &Tensor) -> Result<Tensor> {
        self.target_network.forward(state)
    }

    fn track(&mut self, tau: f64) -> Result<()> {
        track(
            &mut self.varmap,
            &self.vb,
            "target-actor",
            "actor",
            &self.dims,
            tau,
        )
    }
}

#[allow(unused)]
struct Critic<'a> {
    varmap: VarMap,
    vb: VarBuilder<'a>,
    network: Sequential,
    target_network: Sequential,
    size_state: usize,
    size_action: usize,
    dims: Vec<(usize, usize)>,
}

impl Critic<'_> {
    fn new(device: &Device, dtype: DType, size_state: usize, size_action: usize) -> Result<Self> {
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, device);

        let dims: Vec<(usize, usize)> = vec![(size_state + size_action, 400), (400, 300), (300, 1)];

        let make_network = |prefix: &str| {
            let seq = seq()
                .add(linear(
                    dims[0].0,
                    dims[0].1,
                    vb.pp(format!("{prefix}-fc0")),
                )?)
                .add(Activation::Relu)
                .add(linear(
                    dims[1].0,
                    dims[1].1,
                    vb.pp(format!("{prefix}-fc1")),
                )?)
                .add(Activation::Relu)
                .add(linear(
                    dims[2].0,
                    dims[2].1,
                    vb.pp(format!("{prefix}-fc2")),
                )?);
            Ok::<Sequential, Error>(seq)
        };

        let network = make_network("critic")?;
        let target_network = make_network("target-critic")?;

        // this sets the two networks to be equal to each other using tau = 1.0
        track(&mut varmap, &vb, "target-critic", "critic", &dims, 1.0)?;

        Ok(Self {
            varmap,
            vb,
            network,
            target_network,
            size_state,
            size_action,
            dims,
        })
    }

    fn forward(&self, state: &Tensor, action: &Tensor) -> Result<Tensor> {
        let xs = Tensor::cat(&[action, state], 1)?;
        self.network.forward(&xs)
    }

    fn target_forward(&self, state: &Tensor, action: &Tensor) -> Result<Tensor> {
        let xs = Tensor::cat(&[action, state], 1)?;
        self.target_network.forward(&xs)
    }

    fn track(&mut self, tau: f64) -> Result<()> {
        track(
            &mut self.varmap,
            &self.vb,
            "target-critic",
            "critic",
            &self.dims,
            tau,
        )
    }
}

#[allow(unused)]
#[allow(clippy::upper_case_acronyms)]
pub struct DDPG<'a> {
    actor: Actor<'a>,
    actor_optim: AdamW,
    critic: Critic<'a>,
    critic_optim: AdamW,
    gamma: f64,
    tau: f64,
    replay_buffer: ReplayBuffer,
    ou_noise: OuNoise,

    size_state: usize,
    size_action: usize,
    pub train: bool,
}

impl DDPG<'_> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device,
        size_state: usize,
        size_action: usize,
        train: bool,
        actor_lr: f64,
        critic_lr: f64,
        gamma: f64,
        tau: f64,
        buffer_capacity: usize,
        ou_noise: OuNoise,
    ) -> Result<Self> {
        let filter_by_prefix = |varmap: &VarMap, prefix: &str| {
            varmap
                .data()
                .lock()
                .unwrap()
                .iter()
                .filter_map(|(name, var)| name.starts_with(prefix).then_some(var.clone()))
                .collect::<Vec<Var>>()
        };

        let actor = Actor::new(device, DType::F32, size_state, size_action)?;
        let actor_optim = AdamW::new(
            filter_by_prefix(&actor.varmap, "actor"),
            ParamsAdamW {
                lr: actor_lr,
                ..Default::default()
            },
        )?;

        let critic = Critic::new(device, DType::F32, size_state, size_action)?;
        let critic_optim = AdamW::new(
            filter_by_prefix(&critic.varmap, "critic"),
            ParamsAdamW {
                lr: critic_lr,
                ..Default::default()
            },
        )?;

        Ok(Self {
            actor,
            actor_optim,
            critic,
            critic_optim,
            gamma,
            tau,
            replay_buffer: ReplayBuffer::new(buffer_capacity),
            ou_noise,
            size_state,
            size_action,
            train,
        })
    }

    pub fn remember(
        &mut self,
        state: &Tensor,
        action: &Tensor,
        reward: &Tensor,
        next_state: &Tensor,
        terminated: bool,
        truncated: bool,
    ) {
        self.replay_buffer
            .push(state, action, reward, next_state, terminated, truncated)
    }

    pub fn actions(&mut self, state: &Tensor) -> Result<f32> {
        let actions = self
            .actor
            .forward(&state.detach().unsqueeze(0)?)?
            .squeeze(0)?;
        let actions = if self.train {
            (actions + self.ou_noise.sample()?)?
        } else {
            actions
        };
        actions.squeeze(0)?.to_scalar::<f32>()
    }

    pub fn train(&mut self, batch_size: usize) -> Result<()> {
        let (states, actions, rewards, next_states, _, _) =
            match self.replay_buffer.random_batch(batch_size)? {
                Some(v) => v,
                _ => return Ok(()),
            };

        let q_target = self
            .critic
            .target_forward(&next_states, &self.actor.target_forward(&next_states)?)?;
        let q_target = (rewards + (self.gamma * q_target)?.detach())?;
        let q = self.critic.forward(&states, &actions)?;
        let diff = (q_target - q)?;

        let critic_loss = diff.sqr()?.mean_all()?;
        self.critic_optim.backward_step(&critic_loss)?;

        let actor_loss = self
            .critic
            .forward(&states, &self.actor.forward(&states)?)?
            .mean_all()?
            .neg()?;
        self.actor_optim.backward_step(&actor_loss)?;

        self.critic.track(self.tau)?;
        self.actor.track(self.tau)?;

        Ok(())
    }
}

// The impact of the q value of the next state on the current state's q value.
const GAMMA: f64 = 0.99;
// The weight for updating the target networks.
const TAU: f64 = 0.005;
// The capacity of the replay buffer used for sampling training data.
const REPLAY_BUFFER_CAPACITY: usize = 100_000;
// The training batch size for each training iteration.
const TRAINING_BATCH_SIZE: usize = 100;
// The total number of episodes.
const MAX_EPISODES: usize = 100;
// The maximum length of an episode.
const EPISODE_LENGTH: usize = 200;
// The number of training iterations after one episode finishes.
const TRAINING_ITERATIONS: usize = 200;

// Ornstein-Uhlenbeck process parameters.
const MU: f64 = 0.0;
const THETA: f64 = 0.15;
const SIGMA: f64 = 0.1;

const ACTOR_LEARNING_RATE: f64 = 1e-4;
const CRITIC_LEARNING_RATE: f64 = 1e-3;

pub fn run() -> Result<()> {
    let env = GymEnv::new("Pendulum-v1")?;
    println!("action space: {}", env.action_space());
    println!("observation space: {:?}", env.observation_space());

    let size_state = env.observation_space().iter().product::<usize>();
    let size_action = env.action_space();

    let mut agent = DDPG::new(
        &Device::Cpu,
        size_state,
        size_action,
        true,
        ACTOR_LEARNING_RATE,
        CRITIC_LEARNING_RATE,
        GAMMA,
        TAU,
        REPLAY_BUFFER_CAPACITY,
        OuNoise::new(MU, THETA, SIGMA, size_action)?,
    )?;

    let mut rng = rand::rng();

    for episode in 0..MAX_EPISODES {
        // let mut state = env.reset(episode as u64)?;
        let mut state = env.reset(rng.random::<u64>())?;

        let mut total_reward = 0.0;
        for _ in 0..EPISODE_LENGTH {
            let mut action = 2.0 * agent.actions(&state)?;
            action = action.clamp(-2.0, 2.0);

            let step = env.step(vec![action])?;
            total_reward += step.reward;

            agent.remember(
                &state,
                &Tensor::new(vec![action], &Device::Cpu)?,
                &Tensor::new(vec![step.reward as f32], &Device::Cpu)?,
                &step.state,
                step.terminated,
                step.truncated,
            );

            if step.terminated || step.truncated {
                break;
            }
            state = step.state;
        }

        println!("episode {episode} with total reward of {total_reward}");

        for _ in 0..TRAINING_ITERATIONS {
            agent.train(TRAINING_BATCH_SIZE)?;
        }
    }

    println!("Testing...");
    agent.train = false;
    for episode in 0..10 {
        // let mut state = env.reset(episode as u64)?;
        let mut state = env.reset(rng.random::<u64>())?;
        let mut total_reward = 0.0;
        for _ in 0..EPISODE_LENGTH {
            let mut action = 2.0 * agent.actions(&state)?;
            action = action.clamp(-2.0, 2.0);

            let step = env.step(vec![action])?;
            total_reward += step.reward;

            if step.terminated || step.truncated {
                break;
            }
            state = step.state;
        }
        println!("episode {episode} with total reward of {total_reward}");
    }
    Ok(())
}
