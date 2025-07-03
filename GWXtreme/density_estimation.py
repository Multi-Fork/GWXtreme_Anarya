import numpy as np
import scipy.stats


class BoundedKDE(scipy.stats.gaussian_kde):
    def __init__(
            self,
            samples, 
            low=None, 
            high=None,
            bw=None, 
            *args, 
            **kwargs
    ):
        """Initialize with the given bounds.  Either ``low`` or
        ``high`` may be ``None`` if the bounds are one-sided.  Extra
        parameters are passed to :class:`gaussian_kde`.

        :param low: array of lower boundaries.

        :param high: array of upper boundaries.

        """
        samples = np.atleast_2d(samples)

        super().__init__(samples.T, bw_method=bw, *args, **kwargs)

        self.low = low
        self.high = high

    def evaluate(self, points):
        """Return an estimate of the density evaluated at the given
        points."""
        points = np.atleast_2d(points)
        assert points.ndim == 2, 'points must be two-dimensional'
        
        points_orig = np.copy(points)
        
        pdf = super().evaluate(points.T)
        if self.low is not None and self.high is not None:
            for i, (low, high) in enumerate(zip(self.low, self.high)):                
                if not np.isneginf(low) and low is not None:
                        points[:,i] = 2.0 * low - points[:,i]
                        pdf += super().evaluate(points.T)
                        points[:,i] = points_orig[:,i]

                if not np.isposinf(high) and  high is not None:
                        points[:,i] = 2.0 * high - points[:,i]
                        pdf += super().evaluate(points.T)
                        points[:,i] = points_orig[:,i]

        return pdf
    
    def resample(self, size: int | None = None, seed=None):
        samples = super().resample(size, seed)
        if self.low is not None and self.high is not None:
            for i, (low, high) in enumerate(zip(self.low, self.high)):
                
                if not np.isneginf(low) and low is not None:
                    samples[i,:][samples[i,:]<low] = 2.*low- samples[i,:][samples[i,:]<low]
                        
                if not np.isposinf(high) and  high is not None:
                    samples[i,:][samples[i,:]>high] = 2.*high- samples[i,:][samples[i,:]>high]

        return samples.T
    

if __name__ == "__main__":
    samples = np.random.normal([4, 4], [1.2, 5.6], [12_000, 2])
    low = samples.min(axis=0)
    high = samples.max(axis=0)
    
    print(samples.shape, low, high, sep='\n')
    
    kde = BoundedKDE(samples, low, high)

    pdf = kde.evaluate(samples)
    print(pdf.shape)

    rsample = kde.resample()
    print(rsample.shape)

    import matplotlib.pyplot as plt
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.scatter(rsample[:, 0], rsample[:, 1])

    plt.savefig('./test.png')
