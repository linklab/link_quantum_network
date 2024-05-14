from netsquid.protocols import LocalProtocol
import netsquid as ns


class WaitProtocol(LocalProtocol):
    def run(self):
        print(f"Starting protocol at {ns.sim_time()}")

        # yield는 생성기 반복자를 만드는 데 사용되며, 이는 특정 함수의 실행을 중간에서 일시 중단하고 필요에 따라 다시 시작할 수 있게 해줍니다.
        # 생성기 함수를 실행하면 첫 yield 표현식에 도달할 때까지 계속 실행되다가, 해당 지점에서 함수의 실행이 일시 중단됩니다.
        # 이때, 함수는 중단된 위치를 기억하고 있습니다.
        # 따라서 함수를 다시 시작하면 중단됐던 부분부터 실행을 재개하고, 다음 yield에 도달하거나 함수가 끝날 때까지 계속 실행됩니다.
        # 이런 특성을 활용하여, EventExpression에 yield를 사용함으로써 시뮬레이션 중 발생하는 특정 이벤트에 반응하는 인라인 콜백을 생성할 수 있습니다.
        # 이벤트 표현식이 트리거되면(대개는 시뮬레이션의 뒷부분에서), 프로토콜이 다시 시작되어 다음 yield 지점이나 함수의 끝까지 실행을 계속합니다. 이러한 방식으로 프로그램은 복잡한 이벤트 기반 로직을 효과적으로 관리할 수 있습니다.
        yield self.await_timer(100)
        print(f"Ending protocol at {ns.sim_time()}")


def main():
    ns.sim_reset()
    protocol = WaitProtocol()
    protocol.start()
    stats = ns.sim_run()

    print(stats)


if __name__ == "__main__":
    main()