"""

<- Do not use stepx
class TargetPropStepX
  def __init__(self, target_learner: LearningMachine):
    self.target_learner = target_learner

  def step_x(self, conn: Conn, state: State) -> Conn:
     target = target_learner(reverse_conn(conn, use_t=True), state.sub('t'))
     y_conn = reverse_conn(conn, use_t=False)
     target_y = target_learner(y_conn.x, state.sub('y'))
     target_learner.step(target_y, state.sub('y'))
     conn.step_x.x = target
     conn.tie_inp()
     return conn

class TargetPropNoisyStepX
  def __init__(self, target_learner: LearningMachine, noise_f):
    self.target_learner = target_learner
    self.noise_f = noise_f

  def step_x(self, conn: Conn, state: State) -> Conn:
  
     noised = noise_f(conn.x)
     combine_conn = combine_reverse(conn, noised, use_t=True)
     target = target_learner(combine_conn.x, state) 

     combine_conn2 = combine_reverse(conn, noised, use_t=False)     
     target_learner(combine_conn2.x, state)
     target_learner.step(reverse_conn(combine_conn2, use_t=True), state)
     conn.step_x.x = target
     conn.tie_inp()
     return conn

     
class TargetLearner
<- allow for the 


class NoiseTargetLearner
<- instead of using the inverse uses noise plus the target


class TargetDXLearner


class TargetDXNoiseLearner


# okay... This is an easier approach
step_x() 

target = learner([x, y, t]) <- pass in x, y, t for noisy.. [x, t]
learner.step(conn)


# upgrade by TargetLearner <- the base one 

"""


"""

LinearTargetPropLearner()

"""


"""

1. You are giving advice on getting a visa to start a business in Japan
2. What decisions need to be made when getting a visa
3. What do you need to know to make each decision to help the person
  What visas do you need to know that information for. What questions can you ask to get
  information on each of them?
5. List the information surrounding the 
6. when there is enough informaion -> make a recommendation, give a recommendation


4. What visas are available?
- List of 50 categories to consider for each visa


["decisions": [{"": ""}]]





"""