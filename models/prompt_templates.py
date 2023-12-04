# -*- coding: utf-8 -*-
# pylint: disable=too-few-public-methods
"""Sales Support Model (hsr) prompt templates"""

from langchain.prompts import PromptTemplate


class UofPennPromptTemplates:
    """Netec Prompt Templates."""

    sales_role: str = """You are a helpful student advisor at Wharton School of the
        University of Pennsylvania. You provide concise explanations to questions about
        the courses they offer in 100 words or less."""

    @classmethod
    def get_properties(cls):
        """return a list of properties of this class."""
        return [attr for attr in dir(cls) if isinstance(getattr(cls, attr), property)]

    @property
    def online_courses(self) -> PromptTemplate:
        """Get prompt."""
        template = (
            self.sales_role
            + """
        Explain the online courses Wharton offers about {concept}
        """
        )
        return PromptTemplate(input_variables=["concept"], template=template)

    @property
    def certification_programs(self) -> PromptTemplate:
        """Get prompt."""
        template = (
            self.sales_role
            + """
        Summarize their executive and online programs in which learner
        can earns certificates for {concept}
        """
        )
        return PromptTemplate(input_variables=["concept"], template=template)


class NetecPromptTemplates:
    """Netec Prompt Templates."""

    sales_role: str = """You are a helpful sales assistant at Netec who sells
        specialized training and exam preparation services to existing customers.
        You provide concise explanations of the services that Netec offers in 100
        words or less."""

    @classmethod
    def get_properties(cls):
        """return a list of properties of this class."""
        return [attr for attr in dir(cls) if isinstance(getattr(cls, attr), property)]

    @property
    def training_services(self) -> PromptTemplate:
        """Get prompt."""
        template = (
            self.sales_role
            + """
        Explain the training services that Netec offers about {concept}
        """
        )
        return PromptTemplate(input_variables=["concept"], template=template)

    @property
    def oracle_training_services(self) -> PromptTemplate:
        """Get prompt."""
        template = (
            self.sales_role
            + """
        Note that Netec is the exclusive provider in Latin America of Oracle training services
        for the 6 levels of Oracle Certification credentials: Oracle Certified Junior Associate (OCJA),
        Oracle Certified Associate (OCA), Oracle Certified Professional (OCP),
        Oracle Certified Master (OCM), Oracle Certified Expert (OCE) and
        Oracle Certified Specialist (OCS).
        Summarize their programs for {concept}
        """
        )
        return PromptTemplate(input_variables=["concept"], template=template)
