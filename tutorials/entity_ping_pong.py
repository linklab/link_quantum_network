import netsquid as ns
from netsquid import pydynaa


class PingEntity(pydynaa.Entity):
    ping_event_type = pydynaa.EventType("PING_EVENT", "A ping event.")

    def __init__(self):
        self.delay = 10.

    def start(self, qubit):
        # Start the game by scheduling the first ping event after delay
        self.qubit = qubit
        self._schedule_after(self.delay, PingEntity.ping_event_type)

    def wait_for_pong(self, pong_entity):
        # Setup this entity to listen for pong events from a PongEntity
        pong_handler = pydynaa.EventHandler(self._handle_pong_event)
        self._wait(pong_handler, entity=pong_entity, event_type=PongEntity.pong_event_type)

    def _handle_pong_event(self, event):
        # Callback function called by the pong handler when pong event is triggered
        m, prob = ns.qubits.measure(self.qubit, observable=ns.Z)
        labels_z = ("|0>", "|1>")
        print(f"{ns.sim_time():.1f}: Pong event! PingEntity measured "
              f"{labels_z[m]} with probability {prob:.2f}")
        self._schedule_after(self.delay, PingEntity.ping_event_type)


class PongEntity(pydynaa.Entity):
    pong_event_type = pydynaa.EventType("PONG_EVENT", "A pong event.")

    def __init__(self):
        self.delay = 10.

    def wait_for_ping(self, ping_entity):
        # Setup this entity to listen for ping events from a PingEntity
        ping_handler = pydynaa.EventHandler(self._handle_ping_event)
        self._wait(ping_handler, entity=ping_entity, event_type=PingEntity.ping_event_type)

    def _handle_ping_event(self, event):
        # Callback function called by the ping handler when ping event is triggered
        m, prob = ns.qubits.measure(event.source.qubit, observable=ns.X)
        labels_x = ("|+>", "|->")
        print(f"{ns.sim_time():.1f}: Ping event! PongEntity measured "
              f"{labels_x[m]} with probability {prob:.2f}")
        self._schedule_after(self.delay, PongEntity.pong_event_type)


def main():
    # Create entities and register them to each other
    ping = PingEntity()
    pong = PongEntity()
    ping.wait_for_pong(pong)
    pong.wait_for_ping(ping)
    #
    # Create a qubit and instruct the ping entity to start
    qubit, = ns.qubits.create_qubits(1)
    ping.start(qubit)
    run_stats = ns.sim_run(duration=100)
    print(run_stats)


main()
