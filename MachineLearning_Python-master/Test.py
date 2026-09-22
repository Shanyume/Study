from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class KNNRegressor:
    """KNN 回归器封装类。

    使用 StandardScaler 标准化 + KNeighborsRegressor 距离加权回归的管线，
    提供数据校验、模型构建、训练、预测的完整生命周期管理。

    Args:
        n_neighbors (int): K 值，预测时取距离最近的 K 个训练样本。默认为 3。
    """

    def __init__(self, n_neighbors: int = 3):
        """初始化 KNN 回归器。

        Args:
            n_neighbors (int): K 值，预测时取距离最近的 K 个训练样本。默认为 3。
        """
        # K = 3：预测时取距离最近的 3 个训练样本
        self.n_neighbors = n_neighbors
        self._estimator = None

    def validate(self, x_train: list, y_train: list, x_test: list) -> None:
        """校验训练数据和测试数据的基本合法性。

        基本校验：样本数量不足或不一致时提前报错，避免模型内部报出难以定位的错误。

        Args:
            x_train (list): 训练集特征，二维结构。
            y_train (list): 训练集标签。
            x_test (list): 测试集特征，二维结构。

        Raises:
            ValueError: 训练集样本数不足、特征与标签数量不匹配、或测试集为空。
        """
        if len(x_train) < self.n_neighbors:
            raise ValueError("The training set must contain at least n_neighbors samples.")
        if len(x_train) != len(y_train):
            raise ValueError("x_train and y_train must contain the same number of samples.")
        if len(x_test) == 0:
            raise ValueError("x_test must contain at least one sample.")

    def build_pipeline(self):
        """构建标准化 + KNN 回归的管线。

        StandardScaler 对特征做标准化，防止某些特征因数值范围大而主导距离计算；
        weights="distance" 让更近的邻居对预测结果贡献更大。

        Returns:
            Pipeline: scikit-learn 管线对象。
        """
        return make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=self.n_neighbors, weights="distance"),
        )

    def fit(self, x_train: list, y_train: list):
        """训练模型。

        KNN 是惰性学习：fit 只存储训练数据，不做参数拟合。

        Args:
            x_train (list): 训练集特征。
            y_train (list): 训练集标签。

        Returns:
            KNNRegressor: 返回自身，支持链式调用。
        """
        self._estimator = self.build_pipeline()
        self._estimator.fit(x_train, y_train)
        return self

    def predict(self, x_test: list):
        """使用训练好的模型进行预测。

        predict 时才计算距离、找最近邻，并按距离加权平均 K 个邻居的 y 值。

        Args:
            x_test (list): 测试集特征，二维结构。

        Returns:
            ndarray: 预测结果数组。
        """
        return self._estimator.predict(x_test)

    def fit_predict(self, x_train: list, y_train: list, x_test: list):
        """统一入口：校验 → 构建 → 训练 → 预测。

        Args:
            x_train (list): 训练集特征，二维结构。
            y_train (list): 训练集标签。
            x_test (list): 测试集特征，二维结构。

        Returns:
            ndarray: 预测结果数组。
        """
        self.validate(x_train, y_train, x_test)
        self.fit(x_train, y_train)
        return self.predict(x_test)


def main():
    """脚本入口：使用默认数据执行 KNN 回归预测。"""
    # 训练集：4 个样本，每个样本有 3 个特征；y_train 是对应的真实值
    x_train = [[0, 0, 1], [1, 1, 0], [3, 3, 10], [4, 11, 12]]
    y_train = [0.1, 0.2, 0.3, 0.4]

    # 待预测的新样本（必须是二维结构，即使只有一个样本）
    x_test = [[3, 11, 10]]

    regressor = KNNRegressor(n_neighbors=3)
    y_pred = regressor.fit_predict(x_train, y_train, x_test)
    print(y_pred)


if __name__ == "__main__":
    main()
