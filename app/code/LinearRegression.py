import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

class LinearRegression(object):

    """Linear Regression with Gradient Descent and Cross-Validation."""
    
    # We'll use the kfold object from the main script instead,
    # to avoid creating it inside the class.
    # We will keep this here just for default if the user does not provide it.
    kfold = KFold(n_splits=3, shuffle=False)

    def __init__(self,
                 regularization,
                 lr=0.001,
                 method='batch',          # 'batch' | 'mini' | 'sto'
                 num_epochs=1000,         # epochs for training
                 batch_size=50,           # batch size for mini-batch
                 cv=kfold,                # cross-validation object
                 init='zeros',            # weight initialization: 'zeros' | 'xavier'
                 use_momentum=False,      # momentum: T/F
                 momentum=0.9             # momentum values in (0,1)
                 ):
        self.lr             = lr
        self.num_epochs     = num_epochs
        self.batch_size     = batch_size
        self.method         = method
        self.cv             = cv
        self.regularization = regularization

        self.init          = init
        self.use_momentum  = use_momentum
        self.momentum      = momentum
        
        # public attributes to store training history for external use
        self.kfold_scores = []
        self.kfold_r2     = []
        self.theta = None # Initialize theta to None, it will be set in _init_theta

    def mse(self, ytrue, ypred):
        return ((ypred - ytrue) ** 2).sum() / ytrue.shape[0]

    def r2(self, ytrue, ypred):
        ss_res = np.sum((ytrue - ypred) ** 2)
        ss_tot = np.sum((ytrue - np.mean(ytrue)) ** 2)
        return 1.0 - ss_res / (ss_tot + 1e-12)

    def add_intercept(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X]

    def _init_theta(self, n_features):
        """
        Initializes weights according to 'zeros' or 'xavier'.
        
        Params
        ------
        n_features: int
            Number of features (input dimensions).
        """
        # the number of inputs (m) is n_features - 1, since the first column is the bias.
        m = n_features - 1
        
        if self.init == 'xavier':
            # xavier uniform initialization: U[-1/sqrt(m), 1/sqrt(m)]
            limit = 1.0 / np.sqrt(m)
            self.theta = np.random.uniform(-limit, limit, size=(n_features,))
        else: # zeros initialization
            self.theta = np.zeros(n_features)

        # Initialize momentum buffer
        self._vtheta = np.zeros_like(self.theta)

    def fit(self, X_train, y_train):
        # Reset scores for each new fit call
        self.kfold_scores = []
        self.kfold_r2 = []
        
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X_train)):
            X_cross_train = X_train[train_idx]
            y_cross_train = y_train[train_idx]
            X_cross_val   = X_train[val_idx]
            y_cross_val   = y_train[val_idx]
            
            # 1. Add intercept column to both training and validation sets
            X_cross_train = self.add_intercept(X_cross_train)
            X_cross_val   = self.add_intercept(X_cross_val)
            
            # 2. Scale only the feature columns (all but the first)
            scaler = StandardScaler()
            X_cross_train[:, 1:] = scaler.fit_transform(X_cross_train[:, 1:])
            X_cross_val[:, 1:]   = scaler.transform(X_cross_val[:, 1:])

            # Initialize weights
            self._init_theta(X_cross_train.shape[1])

            # CRITICAL CHANGE: The early stopping check has been removed.
            # The loop will now always run for the full number of epochs.
            for epoch in range(self.num_epochs):
                # Shuffle each epoch for SGD and mini-batch
                perm = np.random.permutation(X_cross_train.shape[0])
                Xtr_shuffled = X_cross_train[perm]
                ytr_shuffled = y_cross_train[perm]

                if self.method == 'sto':
                    for i in range(Xtr_shuffled.shape[0]):
                        X_method_train = Xtr_shuffled[i:i+1]
                        y_method_train = ytr_shuffled[i:i+1]
                        train_loss = self._train(X_method_train, y_method_train)
                elif self.method == 'mini':
                    for i in range(0, Xtr_shuffled.shape[0], self.batch_size):
                        X_method_train = Xtr_shuffled[i:i+self.batch_size]
                        y_method_train = ytr_shuffled[i:i+self.batch_size]
                        train_loss = self._train(X_method_train, y_method_train)
                else: # batch
                    train_loss = self._train(Xtr_shuffled, ytr_shuffled)

            yhat_val = self.predict(X_cross_val)
            final_val_loss = self.mse(y_cross_val, yhat_val)
            final_val_r2   = self.r2(y_cross_val, yhat_val)
            
            self.kfold_scores.append(final_val_loss)
            self.kfold_r2.append(final_val_r2)
            
            print(f"Fold {fold}: val_mse={final_val_loss:.6f}  val_r2={final_val_r2:.4f}")

    def _train(self, X, y):
        # Calculate prediction
        yhat = self.predict(X)
        m    = X.shape[0]

        # Calculate gradient for MSE
        grad = (1.0 / m) * X.T @ (yhat - y)
        
        # Add regularization penalty to feature coefficients only (theta[1:])
        reg_derivation = self.regularization.derivation(self.theta[1:])
        grad[1:] += reg_derivation

        if self.use_momentum:
            # Momentum update
            self._vtheta = self.momentum * self._vtheta + self.lr * grad
            self.theta = self.theta - self._vtheta
        else:
            # Vanilla gradient descent
            self.theta = self.theta - self.lr * grad

        return self.mse(y, yhat)

    def predict(self, X):
        """
        Predicts target values for a given input matrix X.
        It assumes X is a matrix of features, possibly without an intercept.
        """
        # Check if the input already has an intercept column.
        # This is the key fix to prevent double-adding the intercept.
        if X.shape[1] + 1 == len(self.theta):
            # If the number of columns is one less than theta's length,
            # it means we need to add an intercept.
            X_with_intercept = self.add_intercept(X)
        elif X.shape[1] == len(self.theta):
            # If the number of columns is the same as theta's length,
            # we assume it already has an intercept.
            X_with_intercept = X
        else:
            # Handle unexpected dimension mismatch gracefully
            raise ValueError(f"Input X has an unexpected number of columns. Expected {len(self.theta)-1} or {len(self.theta)}, but got {X.shape[1]}.")
            
        return X_with_intercept @ self.theta

class NoPenalty:
    """Represents no regularization penalty."""
    def __init__(self, l=0):
        self.l = l
    
    def derivation(self, theta):
        return np.zeros_like(theta)

class RidgePenalty:
    """Represents L2 (Ridge) regularization."""
    def __init__(self, l=0.1):
        self.l = l
    
    def derivation(self, theta):
        return self.l * 2 * theta

class LassoPenalty:
    """Represents L1 (Lasso) regularization."""
    def __init__(self, l=0.1):
        self.l = l
    
    def derivation(self, theta):
        # np.sign returns 0 for 0, which is correct for Lasso's subgradient
        return self.l * np.sign(theta)
