#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "numpy",
#     "pandas",
#     "scipy",
# ]
# ///

"""
Simple A/B testing framework for experiment analysis.
Demonstrates: statistical testing, conversion rate analysis, and confidence intervals.
"""

import numpy as np
import pandas as pd
from scipy import stats

class ABTest:
    def __init__(self, name):
        self.name = name
        self.control_data = []
        self.treatment_data = []

    def add_control(self, conversion):
        """Add control group observation (0 or 1)"""
        self.control_data.append(conversion)

    def add_treatment(self, conversion):
        """Add treatment group observation (0 or 1)"""
        self.treatment_data.append(conversion)

    def calculate_metrics(self):
        """Calculate key metrics for both groups"""
        control = np.array(self.control_data)
        treatment = np.array(self.treatment_data)

        metrics = {
            'control': {
                'n': len(control),
                'conversions': control.sum(),
                'conversion_rate': control.mean() if len(control) > 0 else 0
            },
            'treatment': {
                'n': len(treatment),
                'conversions': treatment.sum(),
                'conversion_rate': treatment.mean() if len(treatment) > 0 else 0
            }
        }

        # Calculate lift
        if metrics['control']['conversion_rate'] > 0:
            lift = ((metrics['treatment']['conversion_rate'] - metrics['control']['conversion_rate'])
                   / metrics['control']['conversion_rate'])
            metrics['lift'] = lift
        else:
            metrics['lift'] = 0

        return metrics

    def run_statistical_test(self, alpha=0.05):
        """Run statistical significance test"""
        control = np.array(self.control_data)
        treatment = np.array(self.treatment_data)

        # Z-test for proportions
        n1, n2 = len(control), len(treatment)
        p1, p2 = control.mean(), treatment.mean()

        # Pooled proportion
        p_pool = (control.sum() + treatment.sum()) / (n1 + n2)

        # Standard error
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))

        # Z-score
        if se > 0:
            z_score = (p2 - p1) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        else:
            z_score = 0
            p_value = 1.0

        # Confidence interval for difference
        diff = p2 - p1
        se_diff = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
        ci_lower = diff - 1.96 * se_diff
        ci_upper = diff + 1.96 * se_diff

        return {
            'z_score': z_score,
            'p_value': p_value,
            'significant': p_value < alpha,
            'alpha': alpha,
            'confidence_interval': (ci_lower, ci_upper)
        }

    def get_report(self):
        """Generate comprehensive report"""
        metrics = self.calculate_metrics()
        test_results = self.run_statistical_test()

        report = f"""
{'=' * 70}
A/B Test Report: {self.name}
{'=' * 70}

SAMPLE SIZES:
  Control:   {metrics['control']['n']:,} users
  Treatment: {metrics['treatment']['n']:,} users

CONVERSIONS:
  Control:   {metrics['control']['conversions']:,} ({metrics['control']['conversion_rate']*100:.2f}%)
  Treatment: {metrics['treatment']['conversions']:,} ({metrics['treatment']['conversion_rate']*100:.2f}%)

PERFORMANCE:
  Lift: {metrics['lift']*100:+.2f}%

STATISTICAL SIGNIFICANCE:
  Z-score:  {test_results['z_score']:.4f}
  P-value:  {test_results['p_value']:.4f}
  Significant at α={test_results['alpha']}: {'YES ✓' if test_results['significant'] else 'NO ✗'}

CONFIDENCE INTERVAL (95%):
  Difference in conversion rate: [{test_results['confidence_interval'][0]*100:.2f}%, {test_results['confidence_interval'][1]*100:.2f}%]

RECOMMENDATION:
  {'Deploy treatment variant - statistically significant improvement!' if test_results['significant'] and metrics['lift'] > 0 else
   'Continue with control - no significant improvement detected' if test_results['significant'] and metrics['lift'] <= 0 else
   'Collect more data - results not yet significant'}
{'=' * 70}
"""
        return report

# Demo: Simulate A/B test
print("A/B Testing Framework Demo")
print("=" * 70)

# Create A/B test
test = ABTest("Homepage Button Color Test")

# Simulate data
np.random.seed(42)

# Control group: 5% conversion rate
control_conversions = np.random.binomial(1, 0.05, 10000)
for conv in control_conversions:
    test.add_control(conv)

# Treatment group: 6% conversion rate (20% lift)
treatment_conversions = np.random.binomial(1, 0.06, 10000)
for conv in treatment_conversions:
    test.add_treatment(conv)

# Generate report
print(test.get_report())

# Additional analysis: Power analysis
print("\nPOWER ANALYSIS:")
print("-" * 70)

def calculate_sample_size(p1, p2, alpha=0.05, power=0.8):
    """Calculate required sample size"""
    # Simplified sample size calculation
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)

    p_avg = (p1 + p2) / 2
    effect_size = abs(p2 - p1)

    if effect_size > 0:
        n = ((z_alpha + z_beta)**2 * 2 * p_avg * (1 - p_avg)) / (effect_size**2)
        return int(np.ceil(n))
    return float('inf')

baseline_rate = 0.05
target_lift = 0.20  # 20% lift
target_rate = baseline_rate * (1 + target_lift)

required_n = calculate_sample_size(baseline_rate, target_rate)
print(f"To detect a {target_lift*100:.0f}% lift with 80% power:")
print(f"  Required sample size per group: {required_n:,} users")
print(f"  Current sample size: {len(test.control_data):,} users")
print(f"  Status: {'✓ Sufficient' if len(test.control_data) >= required_n else '✗ Need more data'}")
