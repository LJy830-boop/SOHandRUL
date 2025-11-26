# 顶部导入与 get_base_models()
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV, BayesianRidge, SGDRegressor
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin, clone
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# XGBoost支持
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
    print("XGBoost可用，将使用GPU加速")
except ImportError:
    XGBRegressor = None
    XGB_AVAILABLE = False
    print("XGBoost不可用")

# CUDA检测（用于XGBoost GPU支持）
def check_cuda_available():
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

CUDA_AVAILABLE = check_cuda_available()

# PyTorch支持（用于神经网络GPU加速）
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        print("PyTorch可用，神经网络将使用GPU加速")
    else:
        print("PyTorch可用但CUDA不可用，神经网络将使用CPU")
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch不可用，使用scikit-learn的MLPRegressor")
    print("PyTorch未安装")

from scipy.optimize import nnls

def get_base_models(random_state: int = 42, cv_splits: int = 10) -> Dict[str, object]:
    """
    获取精简优化的基础模型字典
    原则：6个模型中4个是树模型，相似度过高
    优化：精选4个高度差异化的模型，保留RF+ANN核心模型
    """
    models = {}
    
    # ====== 4个精选基学习器 - 高度差异化 ======
    
    # 1. RF+ANN - 核心混合模型（必须保留）
    models['rf+ann'] = RFPlusANNRegressor(cv=cv_splits, random_state=random_state)
    
    # 2. XGBoost - 梯度提升树（强大且与RF+ANN不同）
    if XGB_AVAILABLE:
        xgb_params = {
            'n_estimators': 1200,
            'max_depth': 6,
            'learning_rate': 0.03,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': random_state,
            'n_jobs': -1,
            'verbosity': 0,
            # 强制模型输出随 'cycle' 特征单调递减
            # 假设 'cycle' 是第一个特征，因此约束为 (-1, 0, 0, ...)
            # 这是一个示例，实际应用中应根据特征顺序调整
            'monotone_constraints': '(-1)' # 假设 'cycle' 是唯一的单调特征，或者通过特征名映射
            # 如果特征顺序已知，可以使用元组，例如：'monotone_constraints': '(-1, 0, 0, 0, 0, 0, 0, 0, 0, 0)'
            # 由于不知道确切的特征数量和顺序，先使用最简形式 (-1) 并在代码中添加注释说明
            
        }
        
        # 如果CUDA可用，添加GPU参数
        if CUDA_AVAILABLE:
            xgb_params.update({
                'tree_method': 'gpu_hist',
                'predictor': 'gpu_predictor'
            })
            print("XGBoost将使用所有可用GPU加速 (gpu_hist)")
        else:
            xgb_params['tree_method'] = 'hist'
            print("XGBoost将使用CPU (hist)")
            
        models['xgb'] = XGBRegressor(**xgb_params)
    
    # 3. 贝叶斯岭回归 - 线性模型（差异化，适合处理相关特征）
    models['bayesian_ridge'] = BayesianRidge(
        alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6
    )
    
    # 4. SVR - 核方法（非线性但原理完全不同）
    models['svr'] = SVR(kernel='rbf', C=10.0, epsilon=0.02, gamma='scale')
    
    # ====== 注释掉的模型（相似度过高，暂时不用）======
    # models['rf'] = ExtraTreesRegressor(...)  # 与RF+ANN相似
    # models['torch_mlp'] = TorchMLPRegressor(...)  # 与RF+ANN中的ANN相似
    # models['ridge'] = RidgeCV(...)  # 与bayesian_ridge相似
    # models['lasso'] = LassoCV(...)  # 与bayesian_ridge相似
    # models['elastic'] = ElasticNetCV(...)  # 与bayesian_ridge相似
    
    return models

def choose_base_models_by_data(X, y, random_state: int = 42, cv_splits: int = 10) -> Dict[str, object]:
    base = get_base_models(random_state, cv_splits)
    n_samples = X.shape[0]
    n_features = X.shape[1]
    missing_frac = float(np.isnan(X.values).mean() if hasattr(X, 'values') else np.isnan(X).mean())
    if n_samples < 500 and 'xgb' in base:
        base['xgb'].set_params(max_depth=6, n_estimators=400, learning_rate=0.07)
        base['rf'].set_params(n_estimators=400)
    if n_features > 200:
        # 仅调参，不再构造内部 StandardScaler 管道
        base['svr'] = SVR(C=5.0, epsilon=0.05, gamma='scale', kernel='rbf')
    if missing_frac > 0.05 and 'xgb' in base:
        base['xgb'].set_params(subsample=0.9, colsample_bytree=0.9)
    
    # 针对大数据集进一步优化
    if n_samples > 1500:
        if 'rf' in base and base['rf'] is not None:
            base['rf'].set_params(
                n_estimators=400,    # 大数据集可以用更多树
                max_depth=25,        # 更深的树
                n_jobs=-1            # 自动使用所有核心
            )
        if 'xgb' in base and base['xgb'] is not None:
            base['xgb'].set_params(
                n_estimators=400,    # 更多轮次
                max_depth=8,         # 更深的树
                n_jobs=-1            # 自动使用所有核心
            )
    
    return base

def build_stacking(base_models: Dict[str, object], final_estimator=None, cv: int = 5) -> StackingRegressor:
    from sklearn.ensemble import StackingRegressor
    if final_estimator is None:
        # 元层：标准化 + ElasticNetCV（轻微L1/L2收缩，稳住强共线）
        final_estimator = Pipeline([
            ('scaler', StandardScaler(with_mean=True)),
            ('enet', ElasticNetCV(
                alphas=np.logspace(-3, 2, 60),
                l1_ratio=[0.05, 0.15, 0.25, 0.5],
                cv=cv,
                max_iter=100000,
                tol=1e-3
            ))
        ])
    stack = StackingRegressor(
        estimators=list(base_models.items()),
        final_estimator=final_estimator,
        passthrough=False,  # 只用基学习器预测作为元层输入
        cv=cv,
        n_jobs=-1
    )
    return stack

class TorchMLPRegressor(BaseEstimator, RegressorMixin):
    """PyTorch实现的GPU加速神经网络回归器"""
    
    def __init__(self, hidden_layer_sizes=(100, 50), learning_rate=0.001, 
                 max_iter=200, batch_size=32, device='auto',
                 weight_decay=0.0, patience=20, min_delta=1e-4, val_split=0.1,
                 verbose=True, log_every=50):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        # 移除内部缩放器，统一由外层管道负责
        self.model = None
        # 补齐属性，避免 get_params/clone 报错
        self.weight_decay = weight_decay
        self.patience = patience
        self.min_delta = min_delta
        self.val_split = val_split
        # 新增：确保属性存在（兼容 sklearn.clone）
        self.verbose = bool(verbose)
        self.log_every = int(log_every)
        # 设备选择：device='auto' 时选择实际设备
        if device == 'auto':
            try:
                import torch as _torch
                self.device = 'cuda' if _torch.cuda.is_available() else 'cpu'
            except Exception:
                self.device = 'cpu'
        else:
            self.device = device
        
        
    def _build_model(self, input_size):
        """构建PyTorch神经网络模型"""
        layers = []
        prev_size = input_size
        
        for hidden_size in self.hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, 1))
        
        return nn.Sequential(*layers).to(self.device)
    
    def fit(self, X, y):
        """训练模型"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch不可用")
        if self.verbose:
            print(f"开始训练PyTorch神经网络，数据形状: {X.shape}")
        
        # 数据验证与更温和的清洗：不整行删除，做填充
        if (np.any(np.isnan(X)) or np.any(np.isinf(X)) or np.any(~np.isfinite(y))):
            if self.verbose:
                print("警告：输入数据包含NaN/Inf，将对X做填充并移除y非有限样本")
            finite_y_mask = np.isfinite(y)
            X = np.asarray(X)[finite_y_mask]
            y = np.asarray(y)[finite_y_mask]
            from sklearn.impute import SimpleImputer
            X = SimpleImputer(strategy='median').fit_transform(X)
            if self.verbose:
                print(f"清理并填充后数据形状: {X.shape}")
        
        if len(X) < 10:
            if self.verbose:
                print("警告：训练数据太少，使用简单模型")
            from sklearn.linear_model import Ridge
            self._fallback_model = Ridge()
            self._fallback_model.fit(X, y)
            self._use_fallback = True
            return self
        
        self._use_fallback = False
        
        # 数据预处理：直接使用外层管道已缩放/清洗后的输入，不再二次缩放
        X_proc = np.asarray(X, dtype=float)
        
        try:
            # 将输入与标签转换为张量（依设备）
            X_tensor = torch.FloatTensor(X_proc).to(self.device)
            y_tensor = torch.FloatTensor(y.reshape(-1, 1)).to(self.device)
        except Exception as e:
            if self.verbose:
                print(f"张量转换失败，回退到CPU: {e}")
            self.device = 'cpu'
            X_tensor = torch.FloatTensor(X_proc).to(self.device)
            y_tensor = torch.FloatTensor(y.reshape(-1, 1)).to(self.device)
        
        # 构建模型
        self.model = self._build_model(X.shape[1])
        
        # 多GPU并行训练支持
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1 and self.device == 'cuda':
            if self.verbose:
                print(f"检测到 {gpu_count} 个GPU，启用DataParallel并行训练")
                print(f"GPU设备列表: {[torch.cuda.get_device_name(i) for i in range(gpu_count)]}")
            self.model = nn.DataParallel(self.model)
            # 调整批次大小以充分利用多GPU
            effective_batch_size = min(self.batch_size * gpu_count, len(X))
            if self.verbose:
                print(f"调整批次大小: {self.batch_size} -> {effective_batch_size} (每GPU: {self.batch_size})")
            self.batch_size = effective_batch_size
        elif self.device == 'cuda':
            if self.verbose:
                print(f"使用单GPU训练: {torch.cuda.get_device_name(0)}")
        else:
            if self.verbose:
                print("使用CPU训练")
        
        # 设置优化器和损失函数（加入权重衰减）
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()
        
        # 创建数据加载器 - 修复pin_memory问题
        dataset = TensorDataset(X_tensor, y_tensor)
        # 当数据已经在GPU上时，不能使用pin_memory
        use_pin_memory = (self.device == 'cuda') and (X_tensor.device.type == 'cpu')
        dataloader = DataLoader(dataset, batch_size=min(self.batch_size, len(X)), shuffle=True, 
                              num_workers=0, pin_memory=use_pin_memory)
        
        if self.verbose:
            print(f"开始训练，总共 {self.max_iter} 个epoch，批次大小: {min(self.batch_size, len(X))}")
        
        # 训练循环
        self.model.train()
        best_loss = float('inf')
        best_state = None
        patience = int(getattr(self, 'patience', 20))
        min_delta = float(getattr(self, 'min_delta', 1e-5))
        val_split = float(getattr(self, 'val_split', 0.1))
        
        for epoch in range(self.max_iter):
            total_loss = 0.0
            batch_count = 0
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                total_loss += loss.item()
                batch_count += 1
            # 验证评估与早停
            if val_split > 0 and len(X_tensor) > 1:
                val_size = max(1, int(len(X_tensor) * val_split))
                with torch.no_grad():
                    val_pred = self.model(X_tensor[-val_size:].to(self.device))
                    val_loss = criterion(val_pred, y_tensor[-val_size:].to(self.device)).item()
                if val_loss < best_loss - min_delta:
                    best_loss = val_loss
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter = patience_counter + 1 if 'patience_counter' in locals() else 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    if best_state is not None:
                        self.model.load_state_dict(best_state)
                    break
            if self.verbose and epoch % int(self.log_every) == 0:
                avg_loss = total_loss / batch_count if batch_count > 0 else 0
                print(f"Epoch {epoch}/{self.max_iter}, Loss: {avg_loss:.6f}")
        if self.verbose:
            print("PyTorch神经网络训练完成")
        return self
    
    def predict(self, X):
        """预测"""
        if hasattr(self, '_use_fallback') and self._use_fallback:
            return self._fallback_model.predict(X)
            
        if not TORCH_AVAILABLE or self.model is None:
            raise ValueError("模型未训练或PyTorch不可用")
        
        # 数据预处理：外层管道已缩放，这里仅做安全转换与裁剪
        if hasattr(X, 'values'):
            X_np = X.values
        else:
            X_np = np.asarray(X)
        X_np = np.nan_to_num(X_np, nan=0.0, posinf=1e6, neginf=-1e6)
        X_np = np.clip(X_np, -1e6, 1e6)
        
        try:
            # 转换为张量并按设备推理
            X_tensor = torch.FloatTensor(X_np).to(self.device)
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X_tensor)
            return predictions.cpu().numpy().flatten()
        except Exception as e:
            print(f"GPU预测失败，尝试CPU: {e}")
            X_tensor = torch.FloatTensor(X_np).to('cpu')
            if hasattr(self.model, 'cpu'):
                model_cpu = self.model.cpu()
            else:
                model_cpu = self.model
            model_cpu.eval()
            with torch.no_grad():
                if isinstance(model_cpu, nn.DataParallel):
                    predictions = model_cpu.module(X_tensor)
                else:
                    predictions = model_cpu(X_tensor)
            return predictions.numpy().flatten()
    
    def get_params(self, deep=True):
        return {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'learning_rate': self.learning_rate,
            'max_iter': self.max_iter,
            'batch_size': self.batch_size,
            'device': self.device,
            'weight_decay': self.weight_decay,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'val_split': self.val_split,
            # 健壮处理，避免旧环境下属性不存在
            'verbose': bool(getattr(self, 'verbose', False)),
            'log_every': int(getattr(self, 'log_every', 50))
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

class BlendingRegressor:
    def __init__(self, base_models: Dict[str, object], cv: int = 5, shrinkage: float = 0.15, core_model: Optional[str] = None, core_prior_weight: float = 0.5, cv_repeats: int = 1):
        self.base_models = base_models
        self.cv = cv
        self.shrinkage = shrinkage
        self.core_model = core_model
        self.core_prior_weight = core_prior_weight
        self.cv_repeats = cv_repeats
    
    def get_params(self, deep=True):
        return {'base_models': self.base_models, 'cv': self.cv, 'shrinkage': self.shrinkage,
                'core_model': self.core_model,
                'core_prior_weight': self.core_prior_weight,
                'cv_repeats': self.cv_repeats}
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y, groups=None):
        # 支持分组，并为每个基模型重新生成split迭代器，避免耗尽
        from sklearn.model_selection import KFold, GroupKFold, RepeatedKFold
        from sklearn.base import clone
        X_np = X.values if hasattr(X, 'values') else np.asarray(X)
        y_np = np.asarray(y)
        model_names = list(self.base_models.keys())
        preds = []
        for name, model in self.base_models.items():
            if groups is not None:
                splitter = GroupKFold(n_splits=self.cv)
                split_iter = splitter.split(X_np, y_np, groups)
                # 单次分组CV（不能重复），按原逻辑生成OOF
                oof = np.zeros_like(y_np, dtype=float)
                for train_idx, test_idx in split_iter:
                    mdl = clone(model)
                    mdl.fit(X_np[train_idx], y_np[train_idx])
                    oof[test_idx] = mdl.predict(X_np[test_idx])
            else:
                # 使用重复K折，提升OOF稳定性
                splitter = RepeatedKFold(n_splits=self.cv, n_repeats=self.cv_repeats, random_state=42)
                oof_sum = np.zeros_like(y_np, dtype=float)
                oof_cnt = np.zeros_like(y_np, dtype=float)
                for train_idx, test_idx in splitter.split(X_np, y_np):
                    mdl = clone(model)
                    mdl.fit(X_np[train_idx], y_np[train_idx])
                    oof_fold = mdl.predict(X_np[test_idx])
                    oof_sum[test_idx] += oof_fold
                    oof_cnt[test_idx] += 1.0
                oof = np.divide(oof_sum, np.maximum(oof_cnt, 1.0))
            preds.append(oof.reshape(-1, 1))
        P = np.hstack(preds)
        w, _ = nnls(P, y_np)
        # 原始NNLS权重（归一化）
        w_norm_raw = w / w.sum() if w.sum() > 0 else w
        w_norm = w_norm_raw.copy()
        # floor：rf+ann 的核心权重下限（归一化前下限0.35）
        if 'rf+ann' in model_names and len(w_norm) > 1:
            core_idx = model_names.index('rf+ann')
            core_min = 0.35
            w_norm[core_idx] = max(w_norm[core_idx], core_min)
            s = w_norm.sum()
            if s > 0:
                w_norm = w_norm / s
        # 使用核心先验进行收缩（避免向均匀收缩拉回0.25）
        if self.core_model is not None and self.core_model in model_names and len(w_norm) > 1:
            prior = np.ones_like(w_norm) * ((1.0 - float(self.core_prior_weight)) / (len(w_norm) - 1))
            prior[model_names.index(self.core_model)] = float(self.core_prior_weight)
        else:
            prior = np.ones_like(w_norm) / len(w_norm)
        if self.shrinkage > 0:
            w_final = (1 - self.shrinkage) * w_norm + self.shrinkage * prior
        else:
            w_final = w_norm
        # 最终保底：保证核心模型至少达到指定占比（例如0.35）
        if self.core_model is not None and self.core_model in model_names:
            core_idx = model_names.index(self.core_model)
            core_min_final = 0.35
            if w_final[core_idx] < core_min_final:
                other_sum = w_final.sum() - w_final[core_idx]
                if other_sum > 0:
                    scale = (1.0 - core_min_final) / other_sum
                    for i in range(len(w_final)):
                        if i != core_idx:
                            w_final[i] *= scale
                    w_final[core_idx] = core_min_final
        self.weights_ = w_final
        print("Blending权重分配 (原始NNLS -> floor -> prior -> final):")
        for name, wn_raw, wn, pr, wf in zip(model_names, w_norm_raw, w_norm, prior, w_final):
            print(f"  {name}: nnls={wn_raw:.6f}, floor={wn:.6f}, prior={pr:.6f}, final={wf:.6f}")
        self.fitted_models_ = {name: clone(model).fit(X_np, y_np) for name, model in self.base_models.items()}
        return self

    def predict(self, X):
        preds = []
        for _, fitted in self.fitted_models_.items():
            preds.append(fitted.predict(X).reshape(-1, 1))
        P = np.hstack(preds)
        return P.dot(self.weights_)

class RFPlusANNRegressor(BaseEstimator, RegressorMixin):
    _estimator_type = 'regressor'

    def __init__(self, rf_params=None, mlp_params=None, cv: int = 5, random_state: int = 42, output_mode: str = 'sum', corr_weight: float = 0.6):
        self.rf_params = rf_params or {
            'n_estimators': 800, 'max_depth': None, 'min_samples_split': 4,
            'min_samples_leaf': 2, 'max_features': 'sqrt', 'bootstrap': True,
            'n_jobs': -1, 'random_state': random_state
        }
        self.mlp_params = mlp_params or {
            'hidden_layer_sizes': (200, 100), 'activation': 'relu', 'solver': 'adam',
            'alpha': 0.0008,
            'learning_rate_init': 0.0012,
            'max_iter': 1200,
            'random_state': random_state,
            'early_stopping': True, 'validation_fraction': 0.12, 'n_iter_no_change': 50
        }
        self.cv = cv
        self.random_state = random_state
        self.output_mode = output_mode
        # 正式超参，满足 clone
        self.corr_weight = corr_weight
    
    def get_params(self, deep=True):
        return {
            'rf_params': self.rf_params,
            'mlp_params': self.mlp_params,
            'cv': self.cv,
            'random_state': self.random_state,
            'output_mode': self.output_mode,
            'corr_weight': self.corr_weight
        }
    
    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    @staticmethod
    def _safe_nan_to_num(arr, nan_val=0.0, pos_val=1e6, neg_val=-1e6):
        arr = np.asarray(arr)
        mask_nan = np.isnan(arr)
        if mask_nan.any():
            arr = arr.copy()
            arr[mask_nan] = nan_val
        mask_inf = np.isinf(arr)
        if mask_inf.any():
            signs = np.sign(arr[mask_inf])
            arr[mask_inf] = np.where(signs >= 0, pos_val, neg_val)
        return arr

    def fit(self, X, y, groups=None):
        # 支持分组OOF，避免信息泄露与索引错位
        X_np = X.values if hasattr(X, 'values') else np.asarray(X)
        y_np = np.asarray(y)
        from sklearn.model_selection import KFold, GroupKFold
        from sklearn.base import clone
        if groups is not None:
            splitter = GroupKFold(n_splits=self.cv)
            split_iter = splitter.split(X_np, y_np, groups)
        else:
            splitter = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            split_iter = splitter.split(X_np, y_np)
        rf = RandomForestRegressor(**self.rf_params)
        oof_rf = np.zeros_like(y_np, dtype=float)
        for train_idx, test_idx in split_iter:
            rf_fold = clone(rf)
            rf_fold.fit(X_np[train_idx], y_np[train_idx])
            oof_rf[test_idx] = rf_fold.predict(X_np[test_idx])
        self.rf_ = rf.fit(X_np, y_np)
        # 残差学习：让 ANN 学习纠正量 residual = y - oof_rf
        oof_rf = np.clip(oof_rf, np.percentile(y_np, 1), np.percentile(y_np, 99))
        oof_rf = RFPlusANNRegressor._safe_nan_to_num(oof_rf, nan_val=float(np.median(y_np)))
        residual = y_np - oof_rf
        # 新增：记录训练期纠正量的99%分位，作为推理时的安全边界
        try:
            self._resid_bound_ = float(np.percentile(np.abs(residual), 99))
            if not np.isfinite(self._resid_bound_) or self._resid_bound_ <= 0:
                self._resid_bound_ = float(np.nanstd(residual) * 3.0) if np.nanstd(residual) > 0 else 1.0
        except Exception:
            self._resid_bound_ = 1.0
        X_enhanced = np.c_[X_np, oof_rf.reshape(-1, 1)]
        X_enhanced = RFPlusANNRegressor._safe_nan_to_num(X_enhanced, nan_val=0.0, pos_val=1e6, neg_val=-1e6)
        # 仅对追加的 oof_rf 列做归一化，避免对外层已缩放的 X 再缩一次
        oof_mean = float(np.nanmean(oof_rf))
        oof_std = float(np.nanstd(oof_rf))
        if not np.isfinite(oof_std) or oof_std <= 0:
            oof_std = 1.0
        self._oof_rf_mean_ = oof_mean
        self._oof_rf_std_ = oof_std
        oof_norm = (oof_rf - oof_mean) / oof_std
        X_enhanced = np.c_[X_np, oof_norm.reshape(-1, 1)]
        X_enhanced = RFPlusANNRegressor._safe_nan_to_num(X_enhanced, nan_val=0.0, pos_val=1e6, neg_val=-1e6)
        mlp = MLPRegressor(**self.mlp_params)
        residual = y_np - oof_rf
        # 鲁棒残差：裁剪异常值，避免 MLP 过拟合噪声
        residual_std = float(np.nanstd(residual))
        if not np.isfinite(residual_std) or residual_std <= 0:
            residual_std = float(np.nanstd(y_np)) if np.nanstd(y_np) > 0 else 1.0
        residual_clip = np.clip(residual, -3 * residual_std, 3 * residual_std)
        self.mlp_ = mlp.fit(X_enhanced, residual_clip)
        try:
            self._y_abs_max_ = float(np.nanmax(np.abs(y_np)))
        except Exception:
            self._y_abs_max_ = None
        return self

    def predict(self, X):
        # 确保数据格式一致
        if hasattr(X, 'values'):
            X_np = X.values
        else:
            X_np = np.asarray(X)
            
        rf_pred = self.rf_.predict(X_np)
        rf_pred = np.clip(rf_pred, -1e6, 1e6)
        rf_pred = RFPlusANNRegressor._safe_nan_to_num(rf_pred, nan_val=0.0)
        # 使用训练期的 oof_rf 归一化参数
        m = float(getattr(self, '_oof_rf_mean_', 0.0))
        s = float(getattr(self, '_oof_rf_std_', 1.0))
        if not np.isfinite(s) or s <= 0:
            s = 1.0
        rf_norm = (rf_pred - m) / s
        X_enhanced = np.c_[X_np, rf_norm.reshape(-1, 1)]
        X_enhanced = RFPlusANNRegressor._safe_nan_to_num(X_enhanced, nan_val=0.0, pos_val=1e6, neg_val=-1e6)
        ann_corr = self.mlp_.predict(X_enhanced)
        if self.output_mode == 'correction':
            bound = getattr(self, '_resid_bound_', None)
            if isinstance(bound, (int, float)) and np.isfinite(bound) and bound > 0:
                ann_corr = np.clip(ann_corr, -bound, bound)
            return ann_corr
        # sum 模式边界与熔断保护
        bound = getattr(self, '_resid_bound_', None)
        if isinstance(bound, (int, float)) and np.isfinite(bound) and bound > 0:
            ann_corr = np.clip(ann_corr, -bound, bound)
        cw = float(getattr(self, 'corr_weight', 0.6))
        pred_sum = rf_pred + cw * ann_corr
        y_abs_max = getattr(self, '_y_abs_max_', None)
        try:
            pred_abs = float(np.nanmax(np.abs(pred_sum)))
            ref_abs = float(np.nanmax(np.abs(rf_pred))) if y_abs_max is None else float(y_abs_max)
        except Exception:
            pred_abs, ref_abs = 0.0, 1.0
        if ref_abs > 0 and pred_abs > 2.0 * ref_abs:
            return rf_pred
        return pred_sum
