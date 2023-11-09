from abc import ABCMeta, abstractmethod


class Index(metaclass=ABCMeta):
    @abstractmethod
    def where_contain(self, loc):
        """
        given a point, return the region it's in
        :param loc: the location of query entry
        :return: region object if found or None
        """
        pass

    @abstractmethod
    def where_intersect(self, bbox):
        """
        given a bbox with format: (1dim_min, 1dim_max, 2dim_min, 2dim_max, ...), return intersected regions
        :param bbox: searching region
        :return: iterable regions
        """
        pass

    @abstractmethod
    def add(self, key, entry) -> None:
        """
        add an entry to its region. Note that add en entry with existed key override the previous.
        :param key: any type.
        :param entry: any.
        :return: None
        """
        pass

