from torch import Tensor

from torchtree.core.utils import process_object, register_class

from .adaptation import Adaptor, DualAveragingStepSize


@register_class
class StanWindowedAdaptation(Adaptor):
    r"""Adapts step size and mass matrix during a warmup period.

    Code adapted from Stan. See online manual for further details
    https://mc-stan.org/docs/reference-manual/hmc-algorithm-parameters.html

    :param step_size_adaptor: step size adaptor
    :param mass_matrix_adaptor: mass matrix adaptor
    :param int num_warmup: number of iteration of warmup period
    :param int init_buffer: width of initial fast adaptation interval
    :param int term_buffer: width of final fast adaptation interval
    :param int base window: initial width of slow adaptation interval
    """

    def __init__(
        self,
        step_size_adaptor: DualAveragingStepSize,
        mass_matrix_adaptor: Adaptor,
        num_warmup: int,
        init_buffer: int,
        term_buffer: int,
        base_window: int,
    ):
        self.num_warmup = 0
        self.adapt_init_buffer = 0
        self.adapt_term_buffer = 0
        self.adapt_base_window = 0
        self.step_size_adaptor = step_size_adaptor
        self.mass_matrix_adaptor = mass_matrix_adaptor

        self._configure_window_parameters(
            num_warmup, init_buffer, term_buffer, base_window
        )

    def restart(self):
        self.adapt_window_counter = 0
        self.adapt_window_size = self.adapt_base_window
        self.adapt_next_window = self.adapt_init_buffer + self.adapt_window_size - 1

    def _configure_window_parameters(
        self, num_warmup, init_buffer, term_buffer, base_window
    ):
        if num_warmup < 20:
            print("WARNING: No estimation is")
            print("         performed for num_warmup < 20")
            exit(1)

        self.num_warmup = num_warmup
        if init_buffer + base_window + term_buffer > num_warmup:
            print("WARNING: There aren't enough warmup iterations to fit the")
            print("         three stages of adaptation as currently configured.")

            self.adapt_init_buffer = 0.15 * num_warmup
            self.adapt_term_buffer = 0.10 * num_warmup
            self.adapt_base_window = num_warmup - (
                self.adapt_init_buffer + self.adapt_term_buffer
            )

            print("         Reducing each adaptation stage to 15%/75%/10% of")
            print("         the given number of warmup iterations:")

            print(f"           init_buffer = {self.adapt_init_buffer}")
            print(f"           adapt_window = {self.adapt_base_window}")
            print(f"           term_buffer = {self.adapt_term_buffer}")
        else:
            self.adapt_init_buffer = init_buffer
            self.adapt_term_buffer = term_buffer
            self.adapt_base_window = base_window
        self.restart()

    def _adaptation_window(self):
        return (
            (self.adapt_window_counter >= self.adapt_init_buffer)
            and (self.adapt_window_counter < self.num_warmup - self.adapt_term_buffer)
            and (self.adapt_window_counter != self.num_warmup)
        )

    def _end_adaptation_window(self):
        return (
            self.adapt_window_counter == self.adapt_next_window
            and self.adapt_window_counter != self.num_warmup
        )

    def _compute_next_window(self):
        if self.adapt_next_window == self.num_warmup - self.adapt_term_buffer - 1:
            return

        self.adapt_window_size *= 2
        self.adapt_next_window = self.adapt_window_counter + self.adapt_window_size

        if self.adapt_next_window == self.num_warmup - self.adapt_term_buffer - 1:
            return

        # Boundary of the following window, not the window just computed
        next_window_boundary = self.adapt_next_window + 2 * self.adapt_window_size

        # If the following window overtakes the full adaptation window,
        # then stretch the current window to the end of the full window
        if next_window_boundary >= self.num_warmup - self.adapt_term_buffer:
            self.adapt_next_window = self.num_warmup - self.adapt_term_buffer - 1

    def learn(self, acceptance_prob: Tensor, sample: int, accepted: bool) -> None:
        if self.adapt_window_counter >= self.num_warmup:
            return

        if self.step_size_adaptor:
            self.step_size_adaptor.learn(acceptance_prob, sample, accepted)

        if self._adaptation_window():
            self.mass_matrix_adaptor.learn(acceptance_prob, sample, accepted)

        if self._end_adaptation_window():
            self.mass_matrix_adaptor.learn(acceptance_prob, sample, accepted)
            if self.step_size_adaptor:
                self.step_size_adaptor.restart()
            self.mass_matrix_adaptor.restart()
            self._compute_next_window()

        self.adapt_window_counter += 1

    @classmethod
    def from_json(cls, data, dic):
        warmup = data["warmup"]
        initial = data["initial_window"]
        final = data["final_window"]
        base = data["base_window"]
        if "step_size_adaptor" in data:
            step_size_adaptor = process_object(data["step_size_adaptor"], dic)
        else:
            step_size_adaptor = None
        mass_matrix_adaptor = process_object(data["mass_matrix_adaptor"], dic)
        return cls(step_size_adaptor, mass_matrix_adaptor, warmup, initial, final, base)
