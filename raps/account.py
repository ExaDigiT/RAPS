"""
Module to capture Account classes Classes:
- Account: representation of an Account
- Accounts: collection of all accounts
"""
import json
from .job import JobStatistics


class Account:
    """Represents an account of a user.

    Each users holds attributes for accounting and statistics, which is used
    for summaries
    Each job consists of various attributes such as the number of nodes required for execution,
    CPU and GPU utilization, wall time, and other relevant parameters (see utils.job_dict).
    The job can transition through different states during its lifecycle, including PENDING,
    RUNNING, COMPLETED, CANCELLED, FAILED, or TIMEOUT.
    """

    def __init__(self, name,
                 priority=0,
                 jobs_enqueued=0,
                 jobs_completed=0,
                 time_allocated=0,
                 energy_allocated=0,
                 avg_power=0,
                 fugaku_points=0
                 ):
        self.name = name
        self.priority = priority
        self.jobs_enqueued = jobs_enqueued
        self.jobs_completed = jobs_completed
        self.time_allocated = time_allocated
        self.energy_allocated = energy_allocated
        self.avg_power = avg_power
        if self.avg_power == 0 and self.energy_allocated != 0:
            self.avg_power = self.time_allocated / self.energy_allocated
        self.fugaku_points = fugaku_points

    def update_fugaku_points(self, average_energy, average_power):
        if average_power == 0:
            self.fugaku_points = 0
        else:
            self.fugaku_points = (average_energy - self.energy_allocated) / average_power

    def update_statistics(self, jobstats, average_user):
        self.jobs_completed += 1
        self.time_allocated += jobstats.run_time * jobstats.num_nodes
        self.energy_allocated += jobstats.energy
        if self.time_allocated == 0:
            self.avg_power = 0
        else:
            self.avg_power = self.energy_allocated / self.time_allocated
        if average_user.avg_power == 0:  # If this is the first job use own power
            average_user.avg_power = self.avg_power
        if average_user.avg_power != 0:  # If no energy was computed no points can be computed.
            self.update_fugaku_points(average_user.energy_allocated, average_user.avg_power)

    def __repr__(self):
        return (f"Account(name={self.name}), "
                f"priority: {self.priority}, "
                f"jobs_enqueued: {self.jobs_enqueued}, "
                f"jobs_completed: {self.jobs_completed}, "
                f"time_allocated: {self.time_allocated}, "
                f"energy_allocated: {self.energy_allocated}, "
                f"avg_power: {self.avg_power}, "
                f"fugaku_points: {self.fugaku_points}, "
                )

    def to_dict(self):
        return {
            "name": self.name,
            "priority": self.priority,
            "jobs_enqueued": self.jobs_enqueued,
            "jobs_completed": self.jobs_completed,
            "time_allocated": self.time_allocated,
            "energy_allocated": self.energy_allocated,
            "avg_power": self.avg_power,
            "fugaku_points": self.fugaku_points
        }

    @classmethod
    def from_dict(acct, account_dict):
        acct = Account(account_dict["name"], priority=account_dict["priority"])
        acct.name = account_dict["name"]
        acct.priority = account_dict["priority"]
        acct.jobs_enqueued = account_dict["jobs_enqueued"]
        acct.jobs_completed = account_dict["jobs_completed"]
        acct.time_allocated = account_dict["time_allocated"]
        acct.energy_allocated = account_dict["energy_allocated"]
        acct.avg_power = account_dict["avg_power"]
        acct.fugaku_points = account_dict["fugaku_points"]
        return acct

    @classmethod
    def merge(cls, account1: 'Account', account2: 'Account') -> 'Account':
        """
        Destructive merge

        Priorities are only set if one is zero or both are equal.
        """
        if account1.name != account2.name:
            raise KeyError(f"{account1.name} != {account2.name}. Input arguments missmatch.")

        merged_account = cls(account1.name)

        if account1.priority == account2.priority:
            merged_account.priority = account1.priority
        elif account1.priority == 0:
            merged_account.priority = account2.priority
        elif account2.priority == 0:
            merged_account.priority = account1.priority
        else:
            raise ValueError("Priority Cannot be derived!")

        merged_account.jobs_enqueued = account1.jobs_enqueued + account2.jobs_enqueued
        merged_account.jobs_completed = account1.jobs_completed + account2.jobs_completed
        merged_account.time_allocated = account1.time_allocated + account2.time_allocated
        merged_account.energy_allocated = account1.energy_allocated + account2.energy_allocated
        if merged_account.energy_allocated != 0:
            merged_account.avg_power = merged_account.time_allocated / merged_account.energy_allocated
        else:
            merged_account.avg_power = 0
        merged_account.fugaku_points = None  # Needs to be invalidated, as averages are not known!

        account1 = None
        account2 = None

        return merged_account


class Accounts:

    def update_average_user(self):
        total_accounts = len(self.account_dict)
        self.average_user.jobs_enqueued = self.all_users.jobs_enqueued / total_accounts
        self.average_user.jobs_completed = self.all_users.jobs_completed / total_accounts
        self.average_user.time_allocated = self.all_users.time_allocated / total_accounts
        self.average_user.energy_allocated = self.all_users.energy_allocated / total_accounts
        self.average_user.avg_power = self.all_users.avg_power / total_accounts
        if self.average_user.jobs_completed != 0.0:
            self.average_user.update_fugaku_points(self.average_user.energy_allocated, self.average_user.avg_power)
        return self

    def __init__(self, jobs=None):
        self.account_dict = dict()
        self.all_users = Account(-2, "All_Users")
        self.average_user = Account(-1, "Avg_User")
        if jobs:
            if not isinstance(jobs, list):
                raise TypeError
            for job_dict in jobs:
                if not isinstance(job_dict, dict):
                    raise TypeError
                if job_dict["account"] not in self.account_dict:
                    self.account_dict[job_dict["account"]] = Account(job_dict["account"], jobs_enqueued=0)
                self.account_dict[job_dict["account"]].jobs_enqueued += 1
                self.all_users.jobs_enqueued += 1
            self.update_average_user()
        pass

    def updates_all_users_by_account(self, account: Account):
        self.all_users.jobs_enqueued += account.jobs_enqueued
        self.all_users.jobs_completed += account.jobs_completed
        self.all_users.time_allocated += account.time_allocated
        self.all_users.energy_allocated += account.energy_allocated
        self.all_users.avg_power = self.energy_allocated / self.time_allocated
        self.update_average_user()  # Only necessary if averag_user was not updated before calling update all users.
        # Therefore As this is needed for fugaku points this should always be called.
        self.all_users.update_fugaku_points(self.average_user.energy_allocated, self.average_user.avg_power)

    def add_account(self, account: Account):
        self.account_dict[account.name] = account
        self.add_user_stats_to_all_users(account)
        # update_average_user() is already called

    @classmethod
    def from_dict(cls, dictionary):
        accounts = cls()

        if 'account_dict' not in dictionary:
            raise KeyError("'account_dict' not in dictionary. Failed to restore.")
        dicts_from_dictionary = dictionary['account_dict']
        accounts.account_dict = {}
        if not isinstance(dicts_from_dictionary, dict):
            raise KeyError("'account_dict' is not a dictionary. Failed to restore.")
        for account_name, account_dict in dicts_from_dictionary.items():
            accounts.account_dict[account_name] = Account.from_dict(account_dict)

        if 'all_users' not in dictionary:
            raise KeyError("'all_users' not in dictionary. Failed to restore.")
        accounts.all_users = Account.from_dict(dictionary['all_users'])

        if 'average_user' not in dictionary:
            raise KeyError("'average_user' not in dictionary. Failed to restore.")
        accounts.average_user = Account.from_dict(dictionary['average_user'])
        return accounts

    @classmethod
    def from_json_filename(cls, filename):
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                json_object = json.load(file)
            return cls.from_dict(json_object)
        except ValueError:
            raise ValueError(f"{file} could not be read using json.load()")

    def update_account_statistics(self, jobstats):
        # Update specific account associated with job
        if isinstance(jobstats, JobStatistics):
            if jobstats.account not in self.account_dict:
                raise ValueError(f"Account {jobstats.account} not registered in Accounts object {self}")
                # self.account_dict[jobstats.account] = Account(jobstats.account)
            account = self.account_dict[jobstats.account]
            account.update_statistics(jobstats, self.average_user)
            self.account_dict[jobstats.account] = account
            # Update the  summary account (all_users) and the average_user account
            self.all_users.update_statistics(jobstats, self.average_user)
            self.update_average_user()

    def to_dict(self):
        acct_dict = {}
        for account_name, account in self.account_dict.items():
            acct_dict[account_name] = account.to_dict()
        ret_dict = {}
        ret_dict['account_dict'] = acct_dict
        ret_dict['all_users'] = self.all_users.to_dict()
        ret_dict['average_user'] = self.average_user.to_dict()
        return ret_dict

    @classmethod
    def merge(cls, accounts1: 'Accounts', accounts2: 'Accounts') -> 'Accounts':
        """
        Destructive merge of accounts
        """
        merged_accounts = Accounts()
        merged_accounts.account_dict = accounts1.account_dict

        for ac2_k, ac2_v in accounts2.account_dict.items():
            if ac2_k in accounts1.account_dict:
                merged_accounts.account_dict[ac2_k] = Account.merge(
                    accounts1.account_dict[ac2_k], accounts2.account_dict[ac2_k])
            else:
                merged_accounts.account_dict[ac2_k] = ac2_v
        for ac1_k, ac1_v in accounts1.account_dict.items():
            if ac1_k not in accounts2.account_dict:
                merged_accounts.account_dict[ac1_k] = ac1_v
            else:
                # Already added above
                pass

        # Update all users -> then update average user -> then fugagku points for all users (order is important!)
        merged_accounts.all_users = Account.merge(accounts1.all_users, accounts2.all_users)
        merged_accounts.update_average_user()
        # Update to average user is needed before fugaku points can be caluculated.
        if merged_accounts.all_users.jobs_completed != 0:
            merged_accounts.all_users.update_fugaku_points(
                merged_accounts.average_user.energy_allocated, merged_accounts.average_user.avg_power)

        for ac_k, ac_v in merged_accounts.account_dict.items():
            if merged_accounts.account_dict[ac_k].jobs_completed != 0:
                merged_accounts.account_dict[ac_k].update_fugaku_points(
                    merged_accounts.average_user.energy_allocated, merged_accounts.average_user.avg_power)

        accounts1 = None
        accounts2 = None
        return merged_accounts
