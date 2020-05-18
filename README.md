# An agent-based model of insect resistance management and mitigation for Bt maize: A social science perspective
by [Yuji Saikai](https://yujisaikai.com), [Terrance M. Hurley](https://www.apec.umn.edu/people/terrance-hurley) & [Paul D. Mitchell](https://aae.wisc.edu/faculty/pdmitchell/)

- Based on [Mesa](https://github.com/projectmesa/mesa)
- Execute ``run.py`` with argument of ``1`` or ``2`` (e.g. ``python run.py 1``)
  - ``1`` for browser visualization
  - ``2`` for detailed specification
- Edit parameters in ``run.py`` for different specification (e.g., landscape size, max steps, random seed, etc.)
- Execute ``batch_policy.py`` for policy experiments
  - It rus parallel using all the CPU cores available
  - Results are saved in ``output``

&nbsp;

**Abstract**

Managing and mitigating agricultural pest resistance to control technologies is a complex system in which biological and social factors spatially and dynamically interact. We build a spatially explicit population genetics model for the evolution of pest resistance to Bt toxins by the insect *Ostrinia nubilalis* and an agent-based model of Bt maize adoption, emphasizing the importance of social factors. The farmer adoption model for Bt maize weighed both individual profitability and adoption decisions of neighboring farmers to mimic the effects of economic incentives and social networks. The model was calibrated using aggregate adoption data for Wisconsin. Simulation experiments with the model provide insights into mitigation policies for a high-dose Bt maize technology once resistance emerges in a pest population. Mitigation policies evaluated include increased refuge requirements for all farms, localized bans on Bt maize where resistance develops, areawide applications of insecticidal sprays on resistant populations, and taxes on Bt maize seed for all farms. Evaluation metrics include resistance allele frequency, pest population density, farmer adoption of Bt maize and economic surplus generated by Bt maize. Based on economic surplus, the results suggest that refuge requirements should remain the foundation of resistance management and mitigation for high-dose Bt maize technologies. For shorter planning horizons (<16 years), resistance mitigation strategies did not improve economic surplus from Bt maize. Social networks accelerated the emergence of resistance, making the optimal policy intervention for longer planning horizons rely more on increased refuge requirements and less on insecticidal sprays targeting resistant pest populations. Overall, the importance social factors play in these results implies more social science research, including agent-based models, would contribute to developing better policies to address the evolution of pest resistance.

[[priprint](btabm.pdf)]
