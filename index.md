## Machine Learning for Epidemiologists: A Glossary of Terms and Applications

You can use the [editor on GitHub](https://github.com/yijinxiang777/Machine-Learning/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Introduction  

Recent years have witnessed a tremendous increase in both interest and use of machine learning (ML) and artificial intelligence (AI) methods in epidemiologic research.[cite a bunch]  Though there are no generally accepted definitions of machine learning or artificial intelligence,[cite Burkov] both tend to focus on or involve identifying relevant patterns in data, or minimizing modeling assumptions.[cite] The fields of ML and AI are immensely complex, and can be implicated in a number of different areas that are relevant to epidemiologic research and practice. These may include predictive analytics, causal inference, automation via specific (versus general) AI, nonparametric (versus parametric) estimation; and supervised, semi-supervised, or unsupervised learning. Because of its increasing popularity, epidemiologists are needing to better understand these fields, their associated complexities, and their relevance or irrelevance to epidemiologic research and practice.   

At present, research on the developments and techniques of machine learning and artificial intelligence are highly technical in nature, and are often difficult for applied epidemiologists and biomedical scientists with little to no training in theoretical or mathematical statistics to access. As a result many advanced techniques in machine learning remain underutilized and/or misunderstood. Furthermore, many of the terms used in ML and AI are also employed in disciplines such as epidemiology, but with slightly or dramatically different meanings.   

The purpose of this paper is to provide an entryway into the complex fields of ML and AI for epidemiologists. Here, we discuss general aspects of the fields of ML and AI relevant to epidemiologists. These include the role and challenges of ML and AI in pattern recognition, prediction, causal discovery, and causal inference. We articulate some of the challenges in the fields of ML and AI, and indicate why they are important to consider when using ML and AI methods in epidemiologic settings. Finally, we provide a glossary of introductory and intermediate terms and concepts to facilitate comprehension, and better integrate ML and AI into epidemiologic research. Importantly, we note that ML and AI methods are not a cure for poor quality data, and emphasize the importance of data integrity, broadly defined, in considering the use of ML and AI techniques in epidemiology.  

### Machine Learning and Artificial Intelligence  

Simply stated, "Artificial intelligence is not intelligence. Machine learning is not learning."[cite Burkov  (2019 p4)] What a typical "learning machine" does is find a math formula which, when applied to collection of inputs (training data), produces outputs [@Burkov2019]. Whether a particular implementation of a machine learning algorithm is successful or not is often marked by how “desirable” the outputs are. The “desirability” of the outputs of a ML algorithm is often defined using precise mathematics. As we discuss in more detail below, without a fundamental understanding of what these mathematical equations imply, researchers can easily misinterpret whether the outputs of a given ML algorithm are meaningfully useful or “optimal” in a practical sense. 
Understanding why these fields are termed “machine learning” and “artificial intelligence”, and the history of their development, may provide some protection against the hype and misplaced expectations of research and application in these areas. Machine learning and artificial intelligence were so named primarily for marketing reasons. For example, Arthur Samuel coined the term “machine learning” in 1959 while at IBM to attract clients and employees with the idea of cutting edge feats of technology.[cite the Mitchell book; is there something in Waldrop Complexity book?; can we find something by Arthur Samuel?]   

ML and AI also developed in different contexts and with different goals. The roots of machine learning originate in the pattern recognition world. Loosely, pattern recognition consists of employing a dataset and an algorithm to discover generalizable patterns that can be used for some productive purpose, such as to inform decision making, for knowledge discovery, or for robotic automation. [example for ML? IBM example?] Artificial intelligence, on the other hand, has its roots in symbolic logic and symbol manipulation. Instead of relying on data, the roots of AI began with computer programs that were written with a particular intent. One of the first examples is the General Problem Solver developed by Herbert Simon and Allen Newell, which consisted of a series of general programmatic if-else statements that would be deployed to find optimal solutions to decision-making problems, such as the “Missionaries and Cannibals” problem or the “prisoner’s dilemma” [see page 22 of @Mitchell2019].
Furthermore, the goals of AI have long been explicit, and are generally divided into two categories: general or strong AI and narrow or specific AI. The objective of general AI is to develop a machine that has patterns of “behavior” similar to that of human beings (e.g., develop, learn, understand, interact). As of yet, researchers have not developed such a machine. However, some researchers anticipate an “AI singularity,” where AI will be able to learn by itself, thus enabling machines to exceed all aspects of human-level intelligence. In contrast, narrow or specific AI, generally refers to machines that are devoted to doing one specific thing well, including, but not limited to, Deep Blue (the IBM chess computer), car manufacturing robots, and virtual assistants (ie. Siri, Cortana, Alexa).  

Despite their distinct origins, today ML and AI share a lot in common. Importantly, there are no generally accepted definitions of machine learning and artificial intelligence. AI is often described as a "branch of computer science that studies the properties of intelligence by synthesizing intelligence" [@AI100]. ML is often described as the "automated detection of meaningful patterns in data"]. Yet, in their modern forms, it is difficult to identify differences between these two areas of research and application. Both areas have been heavily influenced by developments in neural networks and deep learning, while ML or AI based on symbol manipulation are no longer a focus of research and application.

### Machine Learning in Practice: Adapting to Problems/Challenges
Machine Learning and Artificial Intelligence implementations often try to balance two counterposing objectives. On the one hand, machine learning scientists and programmers often rely on variable abstraction to develop generalizable software programs that do not accommodate specific contextual features of the data. On the other hand, the most successful implementations of machine learning have to account for issues or problems that arise when a specific dataset is used to generate an algorithm with a specific purpose.   

For example, Caruana et al (2015) relied on generalized additive models (GAMs) to develop an algorithm for triaging high risk pneumonia patients for hospitalization versus lower risk pneumonia patients for outpatient treatment. However, in its initial implementation, the algorithm predicted that individuals with asthma were less likely to die from pneumonia, compared to non-asthmatics. Additional exploration suggested that this unexpected association was explained by the fact that hospital protocol dictated that asthmatic patients with pneumonia were immediately transferred to the ICU, where they received a higher quality of care compared to the general population. For this reason, asthmatic patients had a lower mortality risk. To address this, the authors incorporated a rule-based logistic regression model that would override their initial GAMs when the training data observation was an asthmatic patient, even if the accuracy of the resulting meta-model was lower than the original ML algorithm. 
Similarly, Zech et al sought to develop deep learning algorithms using radiological data to detect pneumonia in patients recruited at three different hospital groups (i.e., National Institutes of Health Clinical Center (NIH) data, Indiana University Network for Patient Care (IU), and Mount Sinai Hospital (MSH)). Convolutional neural networks (CNNs) were trained using chest x-rays with three different training set combinations, NIH training data, MSH training data, and a combined NIH-MSH training dataset. The authors found much lower prediction accuracy than expected, which was correlated with hospital or department of origin. They found that CNNs became sensitive to features of the hospital or x-ray machine, rather than the patient characteristics encoded in the images. This finding aligns well with other discoveries on the properties of ML for image analysis, including CNNs. Machine learning and AI algorithms can be influenced by minor differences in irrelevant or misleading contextual factors (i.e.,different image processing protocols from different hospitals, devices from different manufacturers used to obtain radiographic images, background noise in the image, etc), or by the prevalence of pneumonia between hospital systems and departments. Essentially, the algorithms relied on contextual features of the images (e.g., resolution and color scheme) that were not relevant to the patients pneumonia status, and which led to poor performance when the algorithms were implemented settings with different contextual features.  

Often, machine learning algorithms are developed to maximize certain features of the algorithm’s performance. For example, researchers often seek to maximize sensitivity and specificity, which is equivalent to maximizing the area under the ROC curve.17 Yet this can lead to a number of false positive predictions. Under resource constrained settings in which clinicians may seek to allocate expensive or invasive treatments, there is a strong motive for minimizing false positive observations. Zheng et al sought to predict the risk of HIV seroconversion in order to more effectively allocate HIV pre-exposure prophylaxis (PrEP) under resource constrained settings. To limit the number of false positive cases, Zheng et al developed a loss function for a family of algorithms that sought to maximize sensitivity subject to constraints on the false positive rate. In effect, Zheng et al incorporated contextual information on the nature of the intervention that would result from the algorithm’s output in order to limit the occurrence of undesirable outcomes (i.e., giving PrEP to HIV-negative individuals who are unlikely to experience seroconversion).  

Each of these three examples demonstrate that one should rarely ever develop and implement a general machine learning algorithm in any given setting. Rather, one should seek to incorporate contextual factors relevant to the research question at hand that may have a bearing on how the algorithm performs. In effect, these examples demonstrate the importance of context and subject matter specifics when constructing ML/AI algorithms.  

### Machine Learning and Artificial Intelligence: Glossary of Terms  

As many of the examples above demonstrate, the successful implementation of any machine learning or artificial intelligence algorithms depends on an awareness of how these methods perform in different settings, and on the options available to researchers for tailoring these methods to the problems and data at hand. To this end, we provide here a contextualized glossary of terms to facilitate understanding of some of the details required to appropriately use ML/AI methods in epidemiologic research. Our glossary is contextualized insofar as we attempt to directly link concepts in ML/AI with epidemiologic concepts and terms. As we will show, relying on many of the core concepts and practices in epidemiology can do much to improve the quality and interpretability of ML/AI implementations. 

* Placeholder for Table of Content (Must not be removed) <newline> {:toc}  
  
  
```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/yijinxiang777/Machine-Learning/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.