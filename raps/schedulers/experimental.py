from typing import List
from enum import Enum
from ..utils import summarize_ranges

from ..policy import BackfillType

# Extending PolicyType:
from ..policy import PolicyType as BasePolicyType
from ..utils import ValueComparableEnum


class ExtendedPolicyType(ValueComparableEnum):
    ACCT_FUGAKU_PTS = 'acct_fugaku_pts'
    ACCT_AVG_P = 'acct_avg_power'
    ACCT_LOW_AVG_P = 'acct_low_avg_power'
    ACCT_AVG_PW4LJ = 'acct_avg_power_w4lj'
    ACCT_EDP = 'acct_edp'
    ACCT_ED2P = 'acct_ed2p'
    ACCT_PDP = 'acct_pdp'


# Boilerplate to combine the enums
combined_members = {
    **{name: member.value for name, member in BasePolicyType.__members__.items()},
    **{name: member.value for name, member in ExtendedPolicyType.__members__.items()}
}
PolicyType = Enum('PolicyType', combined_members, type=ValueComparableEnum)
# The scheduler can now use both the BasePolicies and the Extended Policies


class Scheduler:
    """ Default job scheduler with various scheduling policies. """

    def __init__(self, config, policy, bfpolicy=None, jobs=None, resource_manager=None):
        self.config = config
        if policy is None:  # policy is passed as policy=None, therefore default is not choosen
            policy = "replay"
        self.policy = PolicyType(policy)
        self.bfpolicy = BackfillType(bfpolicy)
        if resource_manager is None:
            raise ValueError("Scheduler requires a ResourceManager instance")
        self.resource_manager = resource_manager
        self.debug = False

    def sort_jobs(self, queue, accounts=None):
        """Sort jobs based on the selected scheduling policy."""
        if self.policy == PolicyType.REPLAY:  # REPLAY NEEDS TO BE THERE
            return sorted(queue, key=lambda job: job.start_time)
        elif self.policy == PolicyType.ACCT_FUGAKU_PTS:
            return self.sort_fugaku_redeeming(queue, accounts)
        elif self.policy == PolicyType.ACCT_AVG_PW4LJ:
            return self.sort_avg_Pw4LJ(queue, accounts)
        elif self.policy == PolicyType.ACCT_AVG_P:
            return self.sort_avg_P(queue, accounts)
        elif self.policy == PolicyType.ACCT_LOW_AVG_P:
            return self.sort_low_avg_P(queue, accounts)
        elif self.policy == PolicyType.ACCT_EDP:
            return self.sort_AEDP(queue, accounts)
        elif self.policy == PolicyType.ACCT_ED2P:
            return self.sort_AED2P(queue, accounts)
        elif self.policy == PolicyType.ACCT_PDP:
            return self.sort_APDP(queue, accounts)
        else:
            raise ValueError(f"Policy not implemented: {self.policy}")

    def schedule(self, queue, running, current_time, accounts=None, sorted=False):
        # Sort the queue in place.
        if not sorted:
            queue[:] = self.sort_jobs(queue, accounts)

        # Iterate over a copy of the queue since we might remove items
        for job in queue[:]:
            if self.policy == PolicyType.REPLAY:
                if job.start_time > current_time:
                    continue  # Replay: Job didn't start yet. Next!
                else:
                    pass
            else:
                pass

            nodes_available = self.check_available_nodes(job)

            if nodes_available:
                self.place_job_and_manage_queues(job, queue, running, current_time)
            else:  # In case the job was not placed, see how we should continue:
                if self.bfpolicy is not None:
                    self.backfill(queue, running, current_time)

                # After backfill dedice continue processing the queue or wait, continuing may result in fairness issues.
                if self.policy in [PolicyType.REPLAY]:  # REPLAY NEEDS TO BE THERE
                    continue  # Regardless if the job at the front of the queue doenst fit, try placing all of them.
                elif self.policy in [PolicyType.ACCT_FUGAKU_PTS,
                                     PolicyType.ACCT_AVG_PW4LJ, PolicyType.ACCT_LOW_AVG_P, PolicyType.ACCT_AVG_P,
                                     PolicyType.ACCT_EDP, PolicyType.ACCT_ED2P, PolicyType.ACCT_PDP
                                     ]:
                    break  # The job at the front of the queue doesnt fit stop processing the queue.
                else:
                    raise NotImplementedError(
                        "Depending on the Policy this choice should be explicit. Add the implementation above!")

    def place_job_and_manage_queues(self, job, queue, running, current_time):
        self.resource_manager.assign_nodes_to_job(job, current_time)
        running.append(job)
        queue.remove(job)
        if self.debug:
            scheduled_nodes = summarize_ranges(job.scheduled_nodes)
            print(f"t={current_time}: Scheduled job {job.id} with time limit "
                  f"{job.time_limit} on nodes {scheduled_nodes}")

    def check_available_nodes(self, job):
        nodes_available = False
        if job.nodes_required <= len(self.resource_manager.available_nodes):
            if self.policy == PolicyType.REPLAY and job.scheduled_nodes:  # Check if we need exact set
                # is exact set available:
                nodes_available = set(job.scheduled_nodes).issubset(set(self.resource_manager.available_nodes))
            else:
                # we dont need the exact set:
                nodes_available = True  # Checked above
                if job.nodes_required == 0:
                    raise ValueError(f"Job Requested zero nodes: {job}")
                # clear scheduled nodes
                job.scheduled_nodes = []
        else:
            pass  # not enough nodes available
        return nodes_available

    def backfill(self, queue: List, running: List, current_time):
        # Try to find a backfill candidate from the entire queue.
        while queue:
            backfill_job = self.find_backfill_job(queue, running, current_time)
            if backfill_job:
                self.place_job_and_manage_queues(backfill_job, queue, running, current_time)
            else:
                break

    def find_backfill_job(self, queue, running, current_time):
        """Finds a backfill job based on available nodes and estimated completion times.

        Loosely based on pseudocode from Leonenkov and Zhumatiy, 'Introducing new backfill-based
        scheduler for slurm resource manager.' Procedia computer science 66 (2015): 661-669.
        """
        if not queue:
            return None

        # Identify when the nex job in the queue could run as a time limit:
        first_job = queue[0]
        nodes_required = 0
        if self.policy == PolicyType.REPLAY and first_job.scheduled_nodes:
            nodes_required = len(first_job.scheduled_nodes)
        else:
            nodes_required = first_job.nodes_required

        sorted_running = sorted(running, key=lambda job: job.end_time)

        # Identify when we have enough nodes therefore the start time of the first_job in line
        shadow_time_end = 0
        shadow_nodes_avail = len(self.resource_manager.available_nodes)
        for job in sorted_running:
            if shadow_nodes_avail >= nodes_required:
                break
            else:
                shadow_nodes_avail += job.nodes_required
                shadow_time_end = job.time_limit

        time_limit = shadow_time_end - current_time
        # We now have the time_limit after which no backfilled job should end
        # as the next job in line has the necessary resrouces after this time limit.

        # Find and return the first job that fits
        if self.bfpolicy == BackfillType.NONE:
            pass
        elif self.bfpolicy == BackfillType.EASY:
            queue[:] = sorted(queue, key=lambda job: job.submit_time)
            return self.return_first_fit(queue, time_limit)
        elif self.bfpolicy == BackfillType.FIRSTFIT:
            pass  # Stay with the prioritization!
            return self.return_first_fit(queue, time_limit)
        elif self.bfpolicy in [BackfillType.BESTFIT,
                               BackfillType.GREEDY,
                               BackfillType.CONSERVATIVE,
                               ]:
            raise NotImplementedError(f"{self.bfpolicy} not implemented! Please implement!")
        else:
            raise NotImplementedError(f"{self.bfpolicy} not implemented.")

    def return_first_fit(self, queue, time_limit):
        for job in queue:
            if job.time_limit <= time_limit:
                nodes_available = self.check_available_nodes(job)
                if nodes_available:
                    return job
                else:
                    continue
            else:
                continue
        return None

    def sort_fugaku_redeeming(self, queue, accounts=None):
        if queue == []:
            return queue
        # Priority queues not yet implemented:
        # Strategy: Sort by Fugaku Points Representing the Priority Queue
        # Everything with negative Fugaku Points get sorted according to normal priority
        priority_triple_list = []
        for job in queue:
            assert accounts and accounts.account_dict
            fugaku_priority = accounts.account_dict[job.account].fugaku_points
            if fugaku_priority is None:
                fugaku_priority = 0
            # Create a tuple of the job and the priority
            priority = job.priority
            priority_triple_list.append((fugaku_priority, priority, job))
        # Sort everythin according to fugaku_points
        priority_triple_list = sorted(priority_triple_list, key=lambda x: x[0], reverse=True)
        # Find the first element with negative fugaku_points
        for cutoff, triple in enumerate(priority_triple_list):
            fugaku_priority, _, _ = triple
            if fugaku_priority < 0:
                break
        first_part = priority_triple_list[:cutoff]
        # Sort everything afterwards according to job priority
        second_part = sorted(priority_triple_list[cutoff:], key=lambda x: x[1], reverse=True)
        queue_a = []
        queue_b = []
        if first_part != []:
            _, _, queue_a = zip(*first_part)
            queue_a = list(queue_a)
        if second_part != []:
            _, _, queue_b = zip(*second_part)
            queue_b = list(queue_b)
        return queue_a + queue_b

    def sort_avg_Pw4LJ(self, queue, accounts=None):
        if queue == []:
            return queue
        priority_tuple_list = []
        for job in queue:
            assert accounts and accounts.account_dict
            power = accounts.account_dict[job.account].avg_power
            if power is None:
                power = 0
            # Create a tuple of the job and the priority
            if job.nodes_required:
                nnodes = job.nodes_required
            elif job.scheduled_nodes:
                nnodes = len(job.scheduled_nodes)
            else:
                raise KeyError("No nodes indicated")

            priority = 100 * nnodes * power
            priority_tuple_list.append((priority, job))
        # Sort everythin according to new priority
        priority_tuple_list = sorted(priority_tuple_list, key=lambda x: x[0], reverse=True)
        queue = []
        if priority_tuple_list != []:
            _, queue = zip(*priority_tuple_list)
            queue = list(queue)
        return queue

    def sort_avg_P(self, queue, accounts=None):
        if queue == []:
            return queue
        priority_tuple_list = []
        for job in queue:
            assert accounts and accounts.accounts_dict
            power = accounts.account_dict[job.account].avg_power
            if power is None:
                power = 0

            priority = power
            priority_tuple_list.append((priority, job))
        # Sort everythin according to power_acct_priority Disregarding size
        priority_tuple_list = sorted(priority_tuple_list, key=lambda x: x[0], reverse=True)
        queue = []
        if priority_tuple_list != []:
            _, queue = zip(*priority_tuple_list)
            queue = list(queue)
        return queue

    def sort_low_avg_P(self, queue, accounts=None):
        if queue == []:
            return queue
        priority_tuple_list = []
        for job in queue:
            assert accounts and accounts.accounts_dict
            power = accounts.account_dict[job.account].avg_power
            if power is None:
                power = 0

            priority = power
            priority_tuple_list.append((priority, job))
        # Sort everythin according to power_acct_priority Disregarding size
        priority_tuple_list = sorted(priority_tuple_list, key=lambda x: x[0], reverse=False)
        queue = []
        if priority_tuple_list != []:
            _, queue = zip(*priority_tuple_list)
            queue = list(queue)
        return queue

    def sort_AEDP(self, queue, accounts=None):
        if queue == []:
            return queue
        priority_tuple_list = []
        for job in queue:
            assert accounts and accounts.accounts_dict
            energy = accounts.account_dict[job.account].energy_allocated
            time = accounts.account_dict[job.account].time_allocated
            if energy is None:
                energy = 0
            if time is None:
                time = 0

            priority = energy * time
            priority_tuple_list.append((priority, job))
        # Sort everythin according to power_acct_priority Disregarding size
        priority_tuple_list = sorted(priority_tuple_list, key=lambda x: x[0], reverse=False)
        queue = []
        if priority_tuple_list != []:
            _, queue = zip(*priority_tuple_list)
            queue = list(queue)
        return queue

    def sort_AED2P(self, queue, accounts=None):
        if queue == []:
            return queue
        priority_tuple_list = []
        for job in queue:
            assert accounts and accounts.accounts_dict
            energy = accounts.account_dict[job.account].energy_allocated
            time = accounts.account_dict[job.account].time_allocated
            if energy is None:
                energy = 0
            if time is None:
                time = 0

            priority = energy * time * time
            priority_tuple_list.append((priority, job))
        # Sort everythin according to power_acct_priority Disregarding size
        priority_tuple_list = sorted(priority_tuple_list, key=lambda x: x[0], reverse=False)
        queue = []
        if priority_tuple_list != []:
            _, queue = zip(*priority_tuple_list)
            queue = list(queue)
        return queue

    def sort_APDP(self, queue, accounts=None):
        if queue == []:
            return queue
        priority_tuple_list = []
        for job in queue:
            assert accounts and accounts.accounts_dict
            power = accounts.account_dict[job.account].avg_power
            time = accounts.account_dict[job.account].time_allocated
            if power is None:
                power = 0
            if time is None:
                time = 0

            priority = power * time
            priority_tuple_list.append((priority, job))
        # Sort everythin according to power_acct_priority Disregarding size
        priority_tuple_list = sorted(priority_tuple_list, key=lambda x: x[0], reverse=False)
        queue = []
        if priority_tuple_list != []:
            _, queue = zip(*priority_tuple_list)
            queue = list(queue)
        return queue
