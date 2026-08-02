# FinSight Demo Guide

A short walkthrough of FinSight’s core workflow: SEC filing ingestion, section-aware retrieval, reranking, grounded answer generation, and citation review.

## Goal

Show how FinSight turns SEC 10-K filings into a citation-grounded research workflow: ingestion, section-aware retrieval, reranking, answer generation, and source verification.

## Setup

Start the app with Docker:

```bash
docker compose up --build