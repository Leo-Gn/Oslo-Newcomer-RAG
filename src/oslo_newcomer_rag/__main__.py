from oslo_newcomer_rag.config import get_settings
from oslo_newcomer_rag.server import run


def main() -> None:
    settings = get_settings()
    run(settings)


if __name__ == "__main__":
    main()
