import logging
import sys
import math
from decimal import Decimal
import random
from time import time
from typing import cast, Dict

from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.actions.Offer import Offer
from geniusweb.actions.PartyId import PartyId
from geniusweb.bidspace.AllBidsList import AllBidsList
from geniusweb.inform.ActionDone import ActionDone
from geniusweb.inform.Finished import Finished
from geniusweb.inform.Inform import Inform
from geniusweb.inform.Settings import Settings
from geniusweb.inform.YourTurn import YourTurn
from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.Domain import Domain
from geniusweb.issuevalue.Value import Value
from geniusweb.issuevalue.ValueSet import ValueSet
from geniusweb.party.Capabilities import Capabilities
from geniusweb.party.DefaultParty import DefaultParty
from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import (
    LinearAdditiveUtilitySpace,
)
from geniusweb.profileconnection.ProfileConnectionFactory import (
    ProfileConnectionFactory,
)
from geniusweb.progress.ProgressTime import ProgressTime
from geniusweb.references.Parameters import Parameters
from tudelft_utilities_logging.ReportToLogger import ReportToLogger

from .utils.opponent_model import OpponentModel


class Group49Agent(DefaultParty):
    """
    Template of a Python geniusweb agent.
    """

    def __init__(self):
        super().__init__()
        self.logger: ReportToLogger = self.getReporter()

        self.domain: Domain = None
        self.parameters: Parameters = None
        self.profile: LinearAdditiveUtilitySpace = None
        self.progress: ProgressTime = None
        self.me: PartyId = None
        self.other: str = None
        self.settings: Settings = None
        self.storage_dir: str = None

        self.last_received_bid: Bid = None
        self.opponent_model: OpponentModel = None
        self.logger.log(logging.INFO, "party is initialized")

        # Average and maximum utility received
        self.maximum_received_utility: Decimal = (Decimal)(-sys.maxsize)
        self.maximum_received_bid: Bid = None
        self.average_received_utility: Decimal = 0
        self.weights: Dict[str, Dict[Value, Decimal]] = {}
        # number of bids our agent has sent
        self.number_of_agent_bids: int = 0
        # number of bids the opponent has sent
        self.number_of_opponent_bids: int = 0
        # last bid send by our agent
        self.last_send_bid: Bid = None
        # concession rate of our agent
        self.our_concession_rate = 0
        self.max_concession_rate = 0
        self.min_concession_rate = 0
        # How much to change concession rate by every time
        self.adapting_rate = 0.001
        self.randomness = 0.05
        self.issue_flexibility = 0.2
        # concession rate opponent
        self.average_concession_rate_opponent = 0
        # list of predicted utilities of opponent from his bids
        self.utilities_opponent_bids = []
        # Bids under this value are not sent nor accepted
        self.reservation_values = []
        # Used for getting towards the Nash product
        self.currProduct = 0


    def notifyChange(self, data: Inform):
        """MUST BE IMPLEMENTED
        This is the entry point of all interaction with your agent after is has been initialised.
        How to handle the received data is based on its class type.

        Args:
            info (Inform): Contains either a request for action or information.
        """

        # a Settings message is the first message that will be send to your
        # agent containing all the information about the negotiation session.
        if isinstance(data, Settings):
            self.settings = cast(Settings, data)
            self.me = self.settings.getID()

            # progress towards the deadline has to be tracked manually through the use of the Progress object
            self.progress = self.settings.getProgress()

            self.parameters = self.settings.getParameters()
            self.storage_dir = self.parameters.get("storage_dir")

            # the profile contains the preferences of the agent over the domain
            profile_connection = ProfileConnectionFactory.create(
                data.getProfile().getURI(), self.getReporter()
            )
            self.profile = profile_connection.getProfile()
            self.domain = self.profile.getDomain()

            profile_connection.close()

        # ActionDone informs you of an action (an offer or an accept)
        # that is performed by one of the agents (including yourself).
        elif isinstance(data, ActionDone):
            action = cast(ActionDone, data).getAction()
            actor = action.getActor()

            # ignore action if it is our action
            if actor != self.me:
                # obtain the name of the opponent, cutting of the position ID.
                self.other = str(actor).rsplit("_", 1)[0]

                # process action done by opponent
                self.opponent_action(action)
        # YourTurn notifies you that it is your turn to act
        elif isinstance(data, YourTurn):
            # execute a turn
            self.my_turn()

        # Finished will be send if the negotiation has ended (through agreement or deadline)
        elif isinstance(data, Finished):
            self.save_data()
            # terminate the agent MUST BE CALLED
            self.logger.log(logging.INFO, "party is terminating:")
            super().terminate()
        else:
            self.logger.log(logging.WARNING, "Ignoring unknown info " + str(data))

    def getCapabilities(self) -> Capabilities:
        """MUST BE IMPLEMENTED
        Method to indicate to the protocol what the capabilities of this agent are.
        Leave it as is for the ANL 2022 competition

        Returns:
            Capabilities: Capabilities representation class
        """
        return Capabilities(
            set(["SAOP"]),
            set(["geniusweb.profile.utilityspace.LinearAdditive"]),
        )

    def send_action(self, action: Action):
        """Sends an action to the opponent(s)

        Args:
            action (Action): action of this agent
        """
        self.getConnection().send(action)

    # give a description of your agent
    def getDescription(self) -> str:
        """MUST BE IMPLEMENTED
        Returns a description of your agent. 1 or 2 sentences.

        Returns:
            str: Agent description
        """
        return "Group 49 agent for the ANL 2022 competition"

    def opponent_action(self, action):
        """Process an action that was received from the opponent.

        Args:
            action (Action): action of opponent
        """
        # if it is an offer, set the last received bid
        if isinstance(action, Offer):
            # create opponent model if it was not yet initialised
            if self.opponent_model is None:
                self.opponent_model = OpponentModel(self.domain)

            bid = cast(Offer, action).getBid()
            bid_value = self.profile.getUtility(bid)

            # update opponent model with bid
            self.opponent_model.update(bid)
            # set bid as last received
            self.last_received_bid = bid
            # add utility of last bid to list of utilities and update the concession rate of the opponent and our agent
            self.utilities_opponent_bids.append(self.opponent_model.get_predicted_utility(bid))
            self.update_opponent_concession_rate()
            self.update_our_concession_rate()
            # increase number of bids received by 1
            self.number_of_opponent_bids += 1

            # Set the new maximum received utility
            if bid_value > self.maximum_received_utility:
                self.maximum_received_utility = bid_value
                self.maximum_received_bid = bid
            
            # Update the average received utility
            if self.average_received_utility == 0:
                self.average_received_utility = bid_value
            else:
                self.average_received_utility = self.add_to_average(self.number_of_opponent_bids, bid_value)

    def my_turn(self):
        """This method is called when it is our turn. It should decide upon an action
        to perform and send this action to the opponent.
        """

        # Start by creating the best bid to propose as offer
        self.update_weights()
        progress = self.progress.get(time() * 1000)
        best_bid = None
        best_bid_score = 0
        best_product = 0
        min_rate = self.min_concession_rate
        if min_rate >= 0: min_rate += 0.025
        max_rate = self.max_concession_rate
        prev_bid = self.last_send_bid
        if prev_bid == None:
            prev_bid = Bid(self.create_first_bid())
        old_utility = 1
        old_utility = self.profile.getUtility(prev_bid)
        for _ in range(50):
            bid = self.find_bid()
            utility = self.profile.getUtility(bid)
            utility_change = utility - old_utility
            product = 0
            if self.last_received_bid == None:
                product = 0
            else:
                #product = self.utilityProduct(bid, self.last_received_bid)
                product = self.nashProduct(bid)
            if utility + product > best_bid_score + best_product and utility_change <= min_rate + self.randomness and utility_change >= max_rate - self.randomness and self.above_reservation_value(bid):
                best_bid_score = utility
                best_bid = bid
                best_product = product
        # Check if the last received offer is good enough
        # If so, accept the offer
        if best_bid == None:
            best_bid = self.last_send_bid
            best_bid_score = old_utility
        if self.maximum_received_bid != None and self.maximum_received_utility > old_utility:
            best_bid = self.maximum_received_bid

        if self.accept_condition(best_bid, self.last_received_bid):
            self.currProduct = best_product
            action = Accept(self.me, self.last_received_bid)
        else:
            higherThanBase = True
            if self.profile.getReservationBid():
                higherThanBase = self.profile.isPreferredOrEqual(self.last_received_bid, self.profile.getReservationBid())
            if higherThanBase and progress > 0.94:
                best_bid = self.maximum_received_bid
            # increase number of bids done by our agent
            self.number_of_agent_bids += 1
            # set last send bid to this bid
            self.last_send_bid = best_bid
            action = Offer(self.me, best_bid)

        # send the action
        self.send_action(action)

    def save_data(self):
        """This method is called after the negotiation is finished. It can be used to store data
        for learning capabilities. Note that no extensive calculations can be done within this method.
        Taking too much time might result in your agent being killed, so use it for storage only.
        """
        data = "Data for learning (see README.md)"
        with open(f"{self.storage_dir}/data.md", "w") as f:
            f.write(data)

    ###########################################################################################
    ################################## Example methods below ##################################
    ###########################################################################################

    def accept_condition(self, our_bid: Bid, last_received_bid: Bid) -> bool:
        if last_received_bid is None:
            return False
        
        # our_bid_value = 0
        # if our_bid is not None:
        our_bid_value = self.profile.getUtility(our_bid)
        opponent_bid_value = self.profile.getUtility(last_received_bid)

        # progress of the negotiation session between 0 and 1 (1 is deadline)
        progress = self.progress.get(time() * 1000)

        # Accept condition: ACconst(β) ∧ (ACtime(0.99) ∨ (ACnext ∧ ACtime(0.5))). So:
        # If the offer is valued above the average/maximum value AND
        # (If 99% of the time towards the deadline has passed OR
        # (The offer is valued above our offers value AND 
        # 50% of the time towards the deadline has pased))

        # Can be changed to maximum received utility instead of average
        # These conditions must always be true
        higherThanBase = True
        if self.profile.getReservationBid():
            higherThanBase = self.profile.isPreferredOrEqual(last_received_bid, self.profile.getReservationBid())

        reservation_conditions = [
            progress > 0.95,
            self.above_reservation_value(last_received_bid) or (higherThanBase and progress > 0.98),
            opponent_bid_value > self.maximum_received_utility - Decimal(0.1)
        ]

        condition = all(reservation_conditions)
        return condition

    def find_bid(self) -> Bid:
        # compose a list of all possible bids
        domain = self.profile.getDomain()
        all_bids = AllBidsList(domain)

        bid: Dict[str, Value] = {}

        # if the first bid call function create_first_bid
        if self.number_of_agent_bids == 0:
            bid = self.create_first_bid()

        # if not the first bid
        else:
            weights = self.weights
            #print(weights)
            #print(weights)
            for issue, values in weights.items():
                random_number = random.random()
                check_number = 0
                #print(values.items())
                #bid[issue] = random.choices(values.keys(), weights=values.values(), k=1)
                for value, weight in values.items():
                    check_number += weight
                    if random_number < check_number:
                        bid[issue] = value
                        #print(value)
                        break
        #print(bid)
        #print(self.profile.getUtility(Bid(bid)))
        return Bid(bid)

    def score_bid(self, bid: Bid, alpha: float = 0.95, eps: float = 0.1) -> float:
        """Calculate heuristic score for a bid

        Args:
            bid (Bid): Bid to score
            alpha (float, optional): Trade-off factor between self interested and
                altruistic behaviour. Defaults to 0.95.
            eps (float, optional): Time pressure factor, balances between conceding
                and Boulware behaviour over time. Defaults to 0.1.

        Returns:
            float: score
        """
        progress = self.progress.get(time() * 1000)

        our_utility = float(self.profile.getUtility(bid))

        time_pressure = 1.0 - progress ** (1 / eps)
        score = alpha * time_pressure * our_utility

        if self.opponent_model is not None:
            opponent_utility = self.opponent_model.get_predicted_utility(bid)
            opponent_score = (1.0 - alpha * time_pressure) * opponent_utility
            score += opponent_score

        return score

    def update_weights(self):
        """Fill randomness values dictionary using all issue weights
        """
        issues = self.profile.getDomain().getIssues()
        for issue in issues:
            values = self.profile.getDomain().getValues(issue)
            total_weight = self.totalWeight(issue)
            for value in values:
                if issue not in self.weights.keys():
                    self.weights[issue] = {}
                if total_weight <= 0:
                    self.weights[issue][value] = 0
                else:
                    weight = self.getWeight(issue, value) / total_weight
                    if weight < 0:
                        self.weights[issue][value] = 0
                    else:
                        self.weights[issue][value] = weight

    def create_first_bid(self) -> Bid:
        """Create the first bid containing all values that will give us the highest utility
        stored in the Preference profile

        Returns:
            Bid: the first bid of our agent
        """
        first_bid: Dict[str, Value] = {}
        domain = self.profile.getDomain()
        value_utilities = self.profile.getUtilities()
        for issue, valueSet in value_utilities.items():
            values = domain.getValues(issue)
            maxUtility = 0
            maxValue = Value("valueA")
            for value in values:
                utility = valueSet.getUtility(value)
                if utility > maxUtility:
                    maxUtility = utility
                    maxValue = value
            first_bid[issue] = maxValue
        self.last_send_bid = Bid(first_bid)
        return first_bid

    def update_opponent_concession_rate(self):
        """Updates the concession rate of the opponent based on the average utility
        """
        utilities = self.utilities_opponent_bids
        utilities = utilities[:-10]
        number_of_utilities = len(utilities)
        sum_of_difference = 0
        for i in range(number_of_utilities - 1):
            utility_difference = 500 * (utilities[i + 1] - utilities[i]) / utilities[i]
            sum_of_difference += utility_difference
        new_concession = 0
        if number_of_utilities > 1:
            new_concession = sum_of_difference / (number_of_utilities - 1)
        self.average_concession_rate_opponent = new_concession

    def update_our_concession_rate(self):
        """Updates the concession rate of our agent based on the alpha and the opponent concession rate
        """
        alpha = self.alpha()
        opponent_rate = self.average_concession_rate_opponent
        #our_concession_rate = 1 / (1 + math.e**(2 * opponent_rate - 1)) - 0.25
        our_concession_rate = alpha * (-opponent_rate + 1)
        if our_concession_rate < 0:
            self.our_concession_rate = 0
        if our_concession_rate > 1:
            self.our_concession_rate = 1
        else:
            self.our_concession_rate = our_concession_rate

        # Min and max concession rates
        progress = self.progress.get(time() * 1000)
        # Opponent based change
        opponent_rate_change = -0.1 * opponent_rate
        # Time based change
        time_passed_change = -0.1 * progress + 0.02
        min_concession_rate = self.min_concession_rate + self.adapting_rate * alpha * (opponent_rate_change + time_passed_change)
        self.min_concession_rate = max(min_concession_rate, -0.025)
        if(min_concession_rate < 0):
            self.max_concession_rate = min(self.min_concession_rate - 0.025, 0)

    def alpha(self):
        return 2 / (1 + math.e**(- self.number_of_opponent_bids / 270.307)) - 1

    def totalWeight(self, issue):
        """Gives the total weight of the two agents models of a specific issue
        """
        domain = self.profile.getDomain()
        values = domain.getValues(issue)
        total_weight = 0
        for value in values:
            total_weight += self.getWeight(issue, value)
        return total_weight

    def getWeight(self, issue, value) -> Decimal:
        """Gives the weight of an value in an issue
        """
        # create opponent model if it was not yet initialised
        if self.opponent_model is None:
            self.opponent_model = OpponentModel(self.domain)
        value_utilities = self.profile.getUtilities()
        issueEstimator = self.opponent_model.issue_estimators.get(issue)
        valueSet = value_utilities.get(issue)
        our_utility = valueSet.getUtility(value)
        opponent_utility = issueEstimator.get_value_utility(value)
        weight = Decimal(1 - self.our_concession_rate) * self.profile.getWeight(issue) * our_utility + Decimal(self.our_concession_rate * issueEstimator.weight * opponent_utility)
        if weight < 0:
            return 0
        else:
            return weight

    def add_to_average(self, size: int, utility: Decimal) -> Decimal:
        return (size * self.average_received_utility + utility) / (size + 1)
    
    def utilityProduct(self, our_bid: Bid, opponent_bid: Bid) -> Decimal:
        return self.profile.getUtility(our_bid) * self.profile.getUtility(opponent_bid)

    def nashProduct(self, bid: Bid) -> Decimal:
        return self.profile.getUtility(bid) * Decimal(self.opponent_model.get_predicted_utility(bid)) 

    def above_reservation_value(self, bid: Bid) -> bool:
        weights = self.profile.getWeights()
        for iss in self.profile.getWeights():
            if self.profile.getUtilities().get(iss).getUtility(bid.getIssueValues()[iss]) - Decimal(self.issue_flexibility) < weights[iss]:
                return False
        return True