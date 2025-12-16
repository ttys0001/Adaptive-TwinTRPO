import os
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
import random
import copy
from datetime import datetime
from tabulate import tabulate
import warnings
import talib
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import shap
warnings.filterwarnings('ignore')
import os
import argparse
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

# 设置随机种子保证实验可重复性
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class FeatureSelector:
    def __init__(self, n_features_to_keep=7):
        self.n_features = n_features_to_keep
        self.selected_features = None

    def select_features(self, X, y):
        # 方法1: 随机森林特征重要性
        rf = RandomForestRegressor(n_estimators=100, random_state=1)
        rf.fit(X, y)
        rf_importance = pd.Series(rf.feature_importances_, index=X.columns)

        # 方法2: XGBoost + SHAP
        xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=1)
        xgb_model.fit(X, y)

        explainer = shap.Explainer(xgb_model)
        shap_values = explainer(X)
        shap_importance = pd.Series(np.abs(shap_values.values).mean(axis=0), index=X.columns)

        # 综合两种方法
        combined_importance = (rf_importance + shap_importance) / 2
        self.selected_features = combined_importance.sort_values(ascending=False).head(self.n_features).index.tolist()
        return self.selected_features

# ==================== 数据预处理类 ====================
class Preprocessor:
    def __init__(self):
        self.feature_selector = FeatureSelector()
        self.feature_names = None

    def compute_technical_indicators(self, df):
        """使用TA-Lib计算技术指标"""
        # 价格数据转换
        open_prices = df['Open'].values.astype(float)
        high_prices = df['High'].values.astype(float)
        low_prices = df['Low'].values.astype(float)
        close_prices = df['Close'].values.astype(float)
        volumes = df['Volume'].values.astype(float)

        # 趋势指标
        df['MA_5'] = talib.MA(close_prices, timeperiod=5)
        df['MA_10'] = talib.MA(close_prices, timeperiod=10)
        df['MA_20'] = talib.MA(close_prices, timeperiod=20)
        df['MA_60'] = talib.MA(close_prices, timeperiod=60)
        df['EMA_12'] = talib.EMA(close_prices, timeperiod=12)
        df['EMA_26'] = talib.EMA(close_prices, timeperiod=26)

        # 动量指标
        df['RSI_14'] = talib.RSI(close_prices, timeperiod=14)
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(close_prices)
        df['ADX_14'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
        df['CCI_14'] = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)
        df['MOM_10'] = talib.MOM(close_prices, timeperiod=10)
        df['ROC_10'] = talib.ROC(close_prices, timeperiod=10)

        # 波动率指标
        df['ATR_14'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
        df['NATR_14'] = talib.NATR(high_prices, low_prices, close_prices, timeperiod=14)
        df['TRANGE'] = talib.TRANGE(high_prices, low_prices, close_prices)

        # 成交量指标
        df['OBV'] = talib.OBV(close_prices, volumes)
        df['AD'] = talib.AD(high_prices, low_prices, close_prices, volumes)
        df['ADOSC'] = talib.ADOSC(high_prices, low_prices, close_prices, volumes, fastperiod=3, slowperiod=10)

        # 其他指标
        df['Bollinger_Upper'], df['Bollinger_Middle'], df['Bollinger_Lower'] = talib.BBANDS(
            close_prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df['SAR'] = talib.SAR(high_prices, low_prices, acceleration=0.02, maximum=0.2)

        # 填充可能的NaN值
        df.dropna(inplace=True) # 删除因计算指标产生的NaN值
        df.reset_index(drop=True, inplace=True)
        return df


    def computeIndicator(self, df, start_date, end_date, feature_selection=True):
        df = df[df['Volume'] != 0]  # 过滤零交易量数据

        # 类型转换
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(float)
        df['Date'] = pd.to_datetime(df['Date'])  # 转换为日期格式



        # 计算技术指标
        df = self.compute_technical_indicators(df)

        # 按时间范围筛选数据
        df = df.loc[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
        df.reset_index(drop=True, inplace=True)  # 重置索引

        # 准备特征和目标变量
        feature_cols = [col for col in df.columns if col not in ['Date']]

        preprocessed_data = df[feature_cols].copy()
        print('Feature columns: ', len(feature_cols), feature_cols)
        X = df[feature_cols]
        y = df['Close'].shift(-1).fillna(method='ffill')  # 预测未来1天的价格
        if feature_selection:
            # 特征选择
            selected_features = self.feature_selector.select_features(X, y)
            print(f"Selected features: {len(selected_features)}, {selected_features}")
            # 提取特征数据和价格趋势
            preprocessed_data = df[selected_features].copy()
            self.feature_names = selected_features

        trend = df[['Date', 'Close']].copy()

        return preprocessed_data, trend


# ==================== 交易环境类 ====================
class TradingEnv:
    def __init__(self, df, startingDate, endingDate, init_money=500000):
        self.n_actions = 3  # 0: hold, 1: buy/open long, 2: sell/close long
        self.window_size = 20  # 状态观测窗口大小
        self.init_money = init_money  # 初始资金
        self.transitionCost = 0.003  # 交易成本
        self.startingDate = startingDate  # 开始日期
        self.endingDate = endingDate  # 结束日期
        self.raw_df = df.copy()
        self.preprocessed_market_data = None
        self.trend = None
        self.account = None
        self.input_size = None
        self.terminal_date = None
        self.t = 0

    def init_market_data(self):
        preprocessor = Preprocessor()
        preprocessed_data, trend_df = preprocessor.computeIndicator(
            self.raw_df, self.startingDate, self.endingDate)
        self.preprocessed_market_data = preprocessed_data
        self.trend = trend_df['Close']

        # 账户表
        self.account = trend_df.copy()
        self.account['Position'] = 0.0  # 1:持仓 0:空仓
        self.account['Action'] = 0.0
        self.account['Q_Action'] = 0.0
        self.account['Holdings'] = 0.0
        self.account['Cash'] = float(self.init_money)
        self.account['Capitals'] = self.account['Holdings'] + self.account['Cash']
        self.account['Returns'] = 0.0

        self.input_size = self.preprocessed_market_data.shape[1]
        self.terminal_date = len(self.trend) - self.window_size - 1

    def reset(self, startingPoint=1):
        if self.preprocessed_market_data is None:
            self.init_market_data()
        self.t = np.clip(startingPoint, self.window_size - 1, self.terminal_date - self.window_size)
        self.account['Position'] = 0.0
        self.account['Action'] = 0.0
        self.account['Q_Action'] = 0.0
        self.account['Holdings'] = 0.0
        self.account['Cash'] = float(self.init_money)
        self.account['Capitals'] = float(self.init_money)
        self.account['Returns'] = 0.0
        self.holdingNum = 0
        self.reward = 0.0
        return self.get_state(self.t)

    def get_state(self, t):
        data_slice = self.preprocessed_market_data.iloc[t + 1 - self.window_size:t + 1, :]
        state = (data_slice - data_slice.mean(axis=0)) / (data_slice.std(axis=0) + 1e-8)
        return np.ravel(state.values)

    def buy_stock(self):
        max_pos = int(
            self.account.loc[self.t - 1, 'Cash'] / (self.account.loc[self.t, 'Close'] * (1 + self.transitionCost)))
        self.holdingNum = max_pos
        self.account.loc[self.t, 'Cash'] = self.account.loc[self.t - 1, 'Cash'] - self.holdingNum * self.account.loc[
            self.t, 'Close'] * (1 + self.transitionCost)
        self.account.loc[self.t, 'Position'] = 1.0
        self.account.loc[self.t, 'Action'] = 1.0

    def sell_stock(self):
        self.account.loc[self.t, 'Cash'] = self.account.loc[self.t - 1, 'Cash'] + self.holdingNum * self.account.loc[
            self.t, 'Close'] * (1 - self.transitionCost)
        self.holdingNum = 0
        self.account.loc[self.t, 'Position'] = 0
        self.account.loc[self.t, 'Action'] = -1.0

    def riskControl(self):
        self.account.loc[self.t, 'Cash'] = self.account.loc[self.t - 1, 'Cash']
        self.account.loc[self.t, 'Action'] = 0.0
        self.account.loc[self.t, 'Position'] = self.account.loc[self.t - 1, 'Position']

    def step(self, action):
        if (action == 1) and (self.account.loc[self.t - 1, 'Position'] == 0):
            self.account.loc[self.t, 'Q_Action'] = 1.0
            self.buy_stock()
        elif action == 2 and (self.account.loc[self.t - 1, 'Position'] == 1):
            self.account.loc[self.t, 'Q_Action'] = -1.0
            self.sell_stock()
        else:
            self.account.loc[self.t, 'Q_Action'] = 0
            self.riskControl()

        # 计算未来5步的收益率变化
        chg = []
        for i in range(5):
            chg.append((self.trend[self.t + i + 1] - self.trend[self.t + i]) / self.trend[self.t + i])

        # 计算索提诺比率
        mean_return = np.mean(chg)
        # 只考虑负收益(下行风险)
        downside_returns = [r for r in chg if r < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
        else:
            downside_std = 0  # 如果没有下行风险，则设为0

        # 索提诺比率 = 平均收益率 / 下行标准差
        if downside_std > 0:
            sortino_ratio = mean_return / downside_std
        else:
            sortino_ratio = mean_return  # 如果没有下行风险，则直接使用平均收益率


        if self.account.loc[self.t, 'Position'] == 1:
            self.reward = sortino_ratio
        else:
            self.reward = 0

        self.account.loc[self.t, 'Holdings'] = self.account.loc[self.t, 'Position'] * self.holdingNum * \
                                               self.account.loc[self.t, 'Close']
        self.account.loc[self.t, 'Capitals'] = self.account.loc[self.t, 'Cash'] + self.account.loc[self.t, 'Holdings']
        self.account.loc[self.t, 'Returns'] = (self.account.loc[self.t, 'Capitals'] - self.account.loc[
            self.t - 1, 'Capitals']) / self.account.loc[self.t - 1, 'Capitals']

        self.t += 1
        done = False
        if self.t == self.terminal_date:
            done = True
            self.account = self.account.drop(index=(self.account.loc[(self.account.index >= self.t)].index))

        next_state = self.get_state(self.t)
        return next_state, self.reward, done

def compute_advantage(gamma, lmbda, td_delta, device):
    td_delta = td_delta.detach().cpu().numpy().flatten()
    advantage = 0.0
    advantages = []
    for delta in reversed(td_delta):
        advantage = delta + gamma * lmbda * advantage
        advantages.append(advantage)
    advantages = torch.tensor(advantages[::-1], dtype=torch.float, device=device)
    return advantages.view(-1, 1)

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class TwinTRPO:
    def __init__(self, state_dim, hidden_dim, action_dim, lmbda,
                 kl_constraint, alpha, critic_lr, gamma, device,
                 vol_alpha=10.0, vol_beta=0.0, rho=0.001):
        """
        新增参数说明:
        vol_alpha, vol_beta: 控制波动率自适应函数 f(Volatility)=1/(1+exp(-vol_alpha*(sigma_t-vol_beta)))
        rho: 用于计算最小步长 eta_min 的比例因子
        """
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.kl_constraint = kl_constraint
        self.alpha = alpha
        self.device = device

        # --- ADDED: volatility-adaptive hyperparams
        self.vol_alpha = vol_alpha
        self.vol_beta = vol_beta
        self.rho = rho
        self.sigma_t = 0.0  # 当前批次波动率（在 update_agent 中计算并赋值）

    def take_action(self, state, greedy=False):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        if greedy:
            return torch.argmax(probs, dim=1).item()
        action_dist = torch.distributions.Categorical(probs)
        return action_dist.sample().item()

    def hessian_matrix_vector_product(self, states, old_action_dists, vector):
        new_action_dists = torch.distributions.Categorical(self.actor(states))
        kl = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists))
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(kl_grad_vector_product, self.actor.parameters())
        grad2_vector = torch.cat([grad.contiguous().view(-1) for grad in grad2])
        return grad2_vector + 0.1 * vector  # 添加阻尼项

    def conjugate_gradient_method(self, grad, states, old_action_dists):
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        rdotr = torch.dot(r, r)
        for i in range(10):
            Hp = self.hessian_matrix_vector_product(states, old_action_dists, p)
            alpha = rdotr / torch.dot(p, Hp)
            x += alpha * p
            r -= alpha * Hp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def compute_surrogate_objective(self, states, actions, advantage, old_log_probs, actor):
        log_probs = torch.log(actor(states).gather(1, actions))
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * advantage)

    def pure_line_search(self, states, actions, advantage, old_log_probs, old_action_dists, dec_vec):
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(self.actor.parameters())
        old_obj = self.compute_surrogate_objective(states, actions, advantage, old_log_probs, self.actor)

        # --- ADDED: 用更准确的二次项估计 u^T H u
        Hp_for_u = self.hessian_matrix_vector_product(states, old_action_dists, dec_vec)
        uHu = torch.dot(dec_vec, Hp_for_u)
        uHu = torch.clamp(uHu, min=1e-8)

        # --- ADDED: 基于当前批次波动率计算自适应缩放因子 f(Volatility)
        try:
            f_vol = 1.0 / (1.0 + math.exp(-self.vol_alpha * (self.sigma_t - self.vol_beta)))
        except OverflowError:
            f_vol = 0.0
        f_vol_torch = torch.tensor(f_vol, dtype=torch.float, device=self.device)

        # --- ADDED: 计算上下界（对应题中公式）
        max_coef = torch.sqrt(2 * self.kl_constraint / (uHu + 1e-8)) * f_vol_torch
        min_coef = (self.rho * torch.abs(old_obj) / (uHu + 1e-8)) * f_vol_torch

        best_para = old_para
        best_obj = old_obj

        for i in range(15):
            coef = max_coef - i * (max_coef - min_coef) / 14
            new_para = old_para + coef * dec_vec
            new_actor = copy.deepcopy(self.actor)
            torch.nn.utils.convert_parameters.vector_to_parameters(new_para, new_actor.parameters())

            new_action_dists = torch.distributions.Categorical(new_actor(states))
            kl_div = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists))
            new_obj = self.compute_surrogate_objective(states, actions, advantage, old_log_probs, new_actor)

            if new_obj > best_obj and kl_div < self.kl_constraint:
                best_para = new_para
                best_obj = new_obj

        return best_para

    def train_policy(self, states, actions, old_action_dists, old_log_probs, advantage):
        surrogate_obj = self.compute_surrogate_objective(states, actions, advantage, old_log_probs, self.actor)
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        descent_direction = self.conjugate_gradient_method(obj_grad, states, old_action_dists)
        new_para = self.pure_line_search(states, actions, advantage, old_log_probs, old_action_dists, descent_direction)
        torch.nn.utils.convert_parameters.vector_to_parameters(new_para, self.actor.parameters())

    def update_agent(self, transition_dict):
        # --- ADDED: 用 transition_dict['rewards'] 的历史波动作为 sigma_t
        try:
            sigma_t = float(np.std(transition_dict.get('rewards', [0.0])))
        except Exception:
            sigma_t = 0.0
        self.sigma_t = sigma_t

        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float, device=self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float, device=self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta, self.device)

        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
        old_action_dists = torch.distributions.Categorical(self.actor(states).detach())

        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.train_policy(states, actions, old_action_dists, old_log_probs, advantage)


# ==================== 性能评估 & 可视化 ====================
class PerformanceEstimator:
    """交易性能评估"""
    def __init__(self, account_df):
        self.window_size = window_size
        self.valid_start = window_size - 1  # 有效数据起始索引
        self.account = account_df.iloc[self.valid_start:]

    def computePnL(self):
        """计算总盈亏"""
        start_date = pd.to_datetime(self.account['Date'].iloc[0])
        end_date = pd.to_datetime(self.account['Date'].iloc[-1])
        self.PnL = self.account['Capitals'].iloc[- 1] - self.account['Capitals'].iloc[0]
        return self.PnL

    def computeCummulatedReturn(self):
        """计算累计收益率"""
        self.CR = ((self.account['Capitals'].iloc[- 1] - self.account['Capitals'].iloc[0]) /
                   self.account['Capitals'].iloc[0]) * 100
        return self.CR

    def computeAnnualizedReturn(self):
        """计算年化收益率"""
        initial_capital = self.account['Capitals'].iloc[0]
        final_capital = self.account['Capitals'].iloc[-1]
        total_return = (final_capital - initial_capital) / initial_capital
        start_date = pd.to_datetime(self.account['Date'].iloc[0])
        end_date = pd.to_datetime(self.account['Date'].iloc[-1])
        days = (end_date - start_date).days
        self.annualizedReturn = 0.0 if days == 0 else 100 * ((1 + total_return) ** (365 / days) - 1)
        return self.annualizedReturn

    def computeAnnualizedVolatility(self):
        """计算年化波动率"""
        self.annualizedVolatility = 100 * np.sqrt(252) * self.account['Returns'].std()
        return self.annualizedVolatility

    def computeSharpeRatio(self, riskFreeRate=0):
        """计算夏普比率"""
        expectedReturn = self.account['Returns'].mean()
        volatility = self.account['Returns'].std()
        self.sharpeRatio = 0 if volatility == 0 else np.sqrt(252) * (expectedReturn - riskFreeRate) / volatility
        return self.sharpeRatio

    def computeMaxDrawdown(self):
        """计算最大回撤"""
        capital = self.account['Capitals'].values
        through = np.argmax(np.maximum.accumulate(capital) - capital)
        if through == 0:
            self.maxDD, self.maxDDD = 0, 0
        else:
            peak = np.argmax(capital[:through])
            self.maxDD = 100 * (capital[peak] - capital[through]) / capital[peak]
            self.maxDDD = through - peak
        return self.maxDD, self.maxDDD

    def computeSortinoRatio(self, riskFreeRate=0):
        """计算Sortino比率"""
        returns = self.account['Returns']
        expectedReturn = returns.mean() - riskFreeRate
        downside_returns = returns[returns < 0]

        downside_std = downside_returns.std()
        if downside_std == 0:
            self.sortinoRatio = 0
        else:
            self.sortinoRatio = np.sqrt(252) * expectedReturn / downside_std
        return self.sortinoRatio

    def computePerformance(self):
        """综合计算所有性能指标"""
        self.computePnL()
        self.computeCummulatedReturn()
        self.computeAnnualizedReturn()
        self.computeAnnualizedVolatility()
        self.computeSharpeRatio()
        self.computeSortinoRatio()
        self.computeMaxDrawdown()
        table = [
            ["PnL", f"{self.PnL:.0f}"],
            ["Cummulated Return", f"{self.CR:.2f}%"],
            ["Annualized Return", f"{self.annualizedReturn:.2f}%"],
            ["Annualized Volatility", f"{self.annualizedVolatility:.2f}%"],
            ["Sharpe Ratio", f"{self.sharpeRatio:.3f}"],
            ["Sortino Ratio", f"{self.sortinoRatio:.3f}"],
            ["Max Drawdown", f"{self.maxDD:.2f}%"],
            ["Max DD Duration", f"{self.maxDDD} days"],
        ]
        return table


class Visualizer:
    def __init__(self, account_df):
        self.window_size = window_size
        self.valid_start = window_size - 1  # 有效数据起始索引
        self.account = account_df.iloc[self.valid_start:]

    """可视化交易结果"""

    def draw_final(self):
        """绘制资金曲线"""
        plt.clf()
        plt.plot(self.account['Capitals'], label='TRPO')
        plt.grid()
        plt.xlabel('Time Step')
        plt.ylabel('Capitals')
        plt.legend()
        plt.savefig(f'{stock_name}/IXIC_TRPO_Capitals.eps', format='eps', dpi=1000, bbox_inches='tight')
        plt.savefig(f'{stock_name}/IXIC_TRPO_Capitals.png', dpi=1000, bbox_inches='tight')

        # plt.show()

    def draw(self):
        """绘制交易信号"""
        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(self.account['Close'], color='royalblue', lw=0.5, label='Price')
        ax1.plot(self.account.loc[self.account['Action'] == 1.0].index,
                 self.account['Close'][self.account['Action'] == 1.0], '^', markersize=6, color='green', label='Buy')
        ax1.plot(self.account.loc[self.account['Action'] == -1.0].index,
                 self.account['Close'][self.account['Action'] == -1.0], 'v', markersize=6, color='red', label='Sell')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Close Price')
        ax1.legend(loc='upper center', ncol=3, frameon=False)
        plt.savefig(f'{stock_name}/IXIC_TRPO_Actions.eps', format='eps', dpi=1000, bbox_inches='tight')
        plt.savefig(f'{stock_name}/IXIC_TRPO_Actions.png', dpi=1000, bbox_inches='tight')
        # plt.show()


# ==================== 训练流程 ====================
def train_with_validation(train_env, test_env, agent, num_episodes=800, test_interval=10):
    """带验证的训练流程"""
    history = {
        'train_returns': [],
        'test_pnl': [],
        'test_returns': [],
        'test_sharpe': [],
        'test_max_dd': []
    }
    best_pnl = -np.inf  # 跟踪最佳测试PnL
    bestModelDir = []

    for i_episode in tqdm(range(num_episodes), desc='Training'):
        # 训练阶段
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        state = train_env.reset()
        done = False
        episode_return = 0
        while not done:
            action = agent.take_action(state)
            next_state, reward, done = train_env.step(action)
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(float(done))
            state = next_state
            episode_return += reward
        agent.update_agent(transition_dict)
        history['train_returns'].append(episode_return)

        # 定期验证
        if (i_episode + 1) >= test_interval:
            test_env.init_market_data()  # 确保每次测试都是独立环境
            state = test_env.reset()
            done = False
            while not done:
                action = agent.take_action(state, greedy=True)
                next_state, _, done = test_env.step(action)
                state = next_state

            # 性能评估
            est = PerformanceEstimator(test_env.account)
            est.computePerformance()
            history['test_pnl'].append(est.PnL)
            history['test_returns'].append(est.CR)
            history['test_sharpe'].append(est.sharpeRatio)
            history['test_max_dd'].append(est.maxDD)
            print("episode", i_episode + 1, "test_pnl", history['test_pnl'][-1], "test_returns",
                  history['test_returns'][-1], "test_sharpe", history['test_sharpe'][- 1])
            # 保存最佳模型
            if est.PnL > best_pnl:
                best_pnl = est.PnL
                best_perf = est.computePerformance()
                torch.save(agent.actor.state_dict(), '{}/TRPO_Best_{}.pth'.format(stock_name, i_episode + 1))
                bestModelDir.append('{}/TRPO_Best_{}.pth'.format(stock_name, i_episode + 1))
                test_env.account.to_csv(f'{stock_name}/best_account_ep{i_episode + 1}_pnl{best_pnl:.0f}.csv', index=False)
                # dra = Visualizer(test_env.account)
                # dra.draw()
    train_env.account.to_csv(f'{stock_name}/train_account.csv', index=False)
    draw_train = Visualizer(train_env.account)
    draw_train.draw_final()
    draw_train.draw()
    print(f'训练完成，测试集最佳 PnL: {best_pnl:.2f}')
    print("\n" + "=" * 40 + " 验证回测结果 " + "=" * 40)
    output_table = tabulate(best_perf, headers=['指标', '值'], tablefmt='fancy_grid')
    print(output_table)
    # 写入到 txt 文件
    # with open('output.txt', 'w', encoding='utf-8') as f:
    #     f.write(output_table)
    file_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
    performance_dict = dict(best_perf)
    df1 = pd.DataFrame([performance_dict])
    df1['model_name'] = file_name_without_extension
    df1['stock_name'] = stock_name
    # save_csv_name = os.path.join(stock_name, f'best_performance_{file_name_without_extension}.csv')
    save_csv_name = f'best_performance_{file_name_without_extension}.csv'
    if os.path.exists(save_csv_name):
        df1.to_csv(save_csv_name, index=False, mode='a', header=False)
    else:
        df1.to_csv(save_csv_name, index=False)
    return history, bestModelDir[-1]


# ==================== 回测函数 ====================
def backtest(model_path, env, greedy):
    """模型回测流程"""
    # 初始化TRPO智能体
    agent = TwinTRPO(state_dim=env.input_size * env.window_size,
                 hidden_dim=hidden_dim,
                 action_dim=env.n_actions,
                 lmbda=lmbda,
                 kl_constraint=kl_constraint,
                 alpha=alpha,
                 critic_lr=critic_lr,
                 gamma=gamma,
                 device=device)

    # 加载最优模型
    agent.actor.load_state_dict(torch.load(model_path))
    agent.actor.eval()

    # 运行回测
    state = env.reset()
    done = False
    while not done:
        action = agent.take_action(state, greedy=greedy)
        next_state, _, done = env.step(action)
        state = next_state

    # 性能评估
    estimator = PerformanceEstimator(env.account)
    estimator.computePerformance()
    perf_table = estimator.computePerformance()
    print("\n" + "=" * 40 + " 回测结果 " + "=" * 40)
    print(tabulate(perf_table, headers=['指标', '值'], tablefmt='fancy_grid'))

    # 可视化结果
    visualizer = Visualizer(env.account)
    visualizer.draw_final()
    visualizer.draw()
    global stock_name
    # 保存详细交易记录
    env.account.to_csv(os.path.join(stock_name, 'backtest_results.csv'), index=False)
    # 附加分析
    print("\n" + "=" * 40 + " 交易分析 " + "=" * 40)
    print(f"总交易次数: {len(env.account[env.account['Action'] != 0])}")
    print(
        f"平均持仓天数: {env.account[env.account['Position'] == 1].shape[0] / len(env.account['Position'].unique()):.1f}")
    print(f"最大单日亏损: {env.account['Returns'].min() * 100:.2f}%")
    print(f"盈利交易占比: {len(env.account[env.account['Returns'] > 0]) / len(env.account) * 100:.1f}%")

    return env.account, perf_table


# ==================== 主程序 ====================
if __name__ == '__main__':
    setup_seed(1)
    parser = argparse.ArgumentParser(description='股票分析程序')
    parser.add_argument('--stocks', default='002230', help='股票代码列表')
    args = parser.parse_args()    
    stock_list = [args.stocks]
    for stock_name in stock_list:
        file_path = f'../Data/{stock_name}.csv'  # 纳斯达克综合指数数据
        if not os.path.exists(stock_name):
            os.mkdir(stock_name)
        df_raw = pd.read_csv(file_path)
        df_raw = df_raw.dropna()  # 删除缺失值
        if 'Adj Close' in df_raw.columns:
            df_raw = df_raw.drop(['Adj Close'], axis=1)  # 移除复权收盘价列

        # 设置训练集和测试集时间范围
        train_startingDate = datetime.strptime('2012-01-01', '%Y-%m-%d')
        train_endingDate   = datetime.strptime('2023-12-31', '%Y-%m-%d')

        test_startingDate  = datetime.strptime('2024-01-01', '%Y-%m-%d')
        test_endingDate    = datetime.strptime('2025-03-31', '%Y-%m-%d')


        # TRPO超参数设置
        hidden_dim = 64  # 神经网络隐藏层维度
        critic_lr = 1e-3  # 价值网络学习率
        lmbda = 0.95  # GAE参数
        kl_constraint = 0.05  # KL散度约束上限
        alpha = 0.5  # 线性搜索衰减系数
        gamma = 0.95  # 折扣因子
        test_interval = 5  # 初始测试间隔
        window_size = 20
        num_episodes = 100
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备选择

        # 初始化环境
        env_train = TradingEnv(df_raw, train_startingDate, train_endingDate)
        env_train.init_market_data()

        # 初始化TRPO智能体
        agent = TwinTRPO(state_dim=env_train.input_size * env_train.window_size,
                    hidden_dim=hidden_dim,
                    action_dim=env_train.n_actions,
                    lmbda=lmbda,
                    kl_constraint=kl_constraint,
                    alpha=alpha,
                    critic_lr=critic_lr,
                    gamma=gamma,
                    device=device)

        # 开始训练
        env_test = TradingEnv(df_raw, test_startingDate, test_endingDate)
        env_test.init_market_data()
        history, BEST_MODEL_PATH = train_with_validation(env_train, env_test, agent, num_episodes, test_interval)

        # 绘制训练曲线
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(history['train_returns'], label='Train Return')
        plt.title('Training Episode Returns')
        plt.legend(framealpha=1)
        # plt.show()
        # BEST_MODEL_PATH = 'TRPO_Best_18.pth'

        # ==================== 执行回测 ====================
        # 运行回测
        account_df, performance = backtest(BEST_MODEL_PATH, env_test, greedy=True)
