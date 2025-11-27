"""Email service for sending notifications."""
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

import structlog

from src.core.config import settings

logger = structlog.get_logger()


class EmailService:
    """Service for sending emails via SMTP."""

    def __init__(self):
        self.smtp_host = getattr(settings, "SMTP_HOST", "localhost")
        self.smtp_port = getattr(settings, "SMTP_PORT", 587)
        self.smtp_user = getattr(settings, "SMTP_USER", None)
        self.smtp_password = getattr(settings, "SMTP_PASSWORD", None)
        self.from_email = getattr(settings, "FROM_EMAIL", "noreply@investai.com.br")
        self.from_name = getattr(settings, "FROM_NAME", "InvestAI")

    def send_email(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: Optional[str] = None,
    ) -> bool:
        """Send an email."""
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = f"{self.from_name} <{self.from_email}>"
            msg["To"] = to_email

            # Add text version
            if text_content:
                text_part = MIMEText(text_content, "plain", "utf-8")
                msg.attach(text_part)

            # Add HTML version
            html_part = MIMEText(html_content, "html", "utf-8")
            msg.attach(html_part)

            # For development/testing - just log the email
            if settings.ENVIRONMENT == "development" and not self.smtp_user:
                logger.info(
                    "Email would be sent (dev mode)",
                    to=to_email,
                    subject=subject,
                )
                return True

            # Send via SMTP
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                if self.smtp_user and self.smtp_password:
                    server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.from_email, to_email, msg.as_string())

            logger.info("Email sent successfully", to=to_email, subject=subject)
            return True

        except Exception as e:
            logger.error("Failed to send email", to=to_email, error=str(e))
            return False


# Global instance
email_service = EmailService()


# Email templates
def get_welcome_email_html(name: str) -> str:
    """Generate welcome email HTML."""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 8px 8px 0 0; }}
            .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 8px 8px; }}
            .button {{ display: inline-block; background: #667eea; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; margin-top: 20px; }}
            .footer {{ text-align: center; margin-top: 20px; color: #666; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Bem-vindo ao InvestAI!</h1>
            </div>
            <div class="content">
                <p>Olá, <strong>{name}</strong>!</p>
                <p>Obrigado por se cadastrar no InvestAI. Estamos muito felizes em ter você conosco!</p>
                <p>Com o InvestAI, você terá acesso a:</p>
                <ul>
                    <li>Análise inteligente do seu portfólio</li>
                    <li>Recomendações personalizadas de investimentos</li>
                    <li>Alertas de preço em tempo real</li>
                    <li>Assistente de IA para tirar suas dúvidas</li>
                </ul>
                <p>Comece agora mesmo preenchendo seu perfil de investidor!</p>
                <a href="https://investai.com.br/onboarding" class="button">Começar Agora</a>
            </div>
            <div class="footer">
                <p>InvestAI - Investimentos inteligentes para todos</p>
            </div>
        </div>
    </body>
    </html>
    """


def get_price_alert_email_html(
    name: str,
    ticker: str,
    condition: str,
    target_price: float,
    current_price: float,
) -> str:
    """Generate price alert email HTML."""
    condition_text = "acima de" if condition == "above" else "abaixo de"
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background: #f59e0b; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }}
            .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 8px 8px; }}
            .alert-box {{ background: white; border: 2px solid #f59e0b; border-radius: 8px; padding: 20px; margin: 20px 0; text-align: center; }}
            .ticker {{ font-size: 24px; font-weight: bold; color: #1f2937; }}
            .price {{ font-size: 32px; font-weight: bold; color: #059669; }}
            .button {{ display: inline-block; background: #667eea; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Alerta de Preço Acionado!</h1>
            </div>
            <div class="content">
                <p>Olá, <strong>{name}</strong>!</p>
                <p>Seu alerta de preço foi acionado:</p>
                <div class="alert-box">
                    <div class="ticker">{ticker}</div>
                    <p>Preço atual: <span class="price">R$ {current_price:.2f}</span></p>
                    <p>Condição: {condition_text} R$ {target_price:.2f}</p>
                </div>
                <p>Acesse sua conta para mais detalhes e tomar uma decisão.</p>
                <a href="https://investai.com.br/portfolio" class="button">Ver Portfólio</a>
            </div>
        </div>
    </body>
    </html>
    """


def get_daily_report_email_html(
    name: str,
    total_value: float,
    daily_change: float,
    daily_change_pct: float,
    top_gainers: list,
    top_losers: list,
) -> str:
    """Generate daily report email HTML."""
    change_color = "#059669" if daily_change >= 0 else "#dc2626"
    change_sign = "+" if daily_change >= 0 else ""

    gainers_html = ""
    for g in top_gainers[:3]:
        gainers_html += f"<li>{g['ticker']}: +{g['change_pct']:.2f}%</li>"

    losers_html = ""
    for l in top_losers[:3]:
        losers_html += f"<li>{l['ticker']}: {l['change_pct']:.2f}%</li>"

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }}
            .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 8px 8px; }}
            .summary-box {{ background: white; border-radius: 8px; padding: 20px; margin: 20px 0; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .value {{ font-size: 28px; font-weight: bold; color: #1f2937; }}
            .change {{ font-size: 18px; color: {change_color}; }}
            .section {{ margin: 20px 0; }}
            .section h3 {{ color: #1f2937; border-bottom: 2px solid #667eea; padding-bottom: 5px; }}
            .gainer {{ color: #059669; }}
            .loser {{ color: #dc2626; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Resumo Diário do Portfólio</h1>
            </div>
            <div class="content">
                <p>Olá, <strong>{name}</strong>! Aqui está seu resumo diário:</p>
                <div class="summary-box">
                    <p>Valor Total do Portfólio</p>
                    <div class="value">R$ {total_value:,.2f}</div>
                    <div class="change">{change_sign}R$ {daily_change:,.2f} ({change_sign}{daily_change_pct:.2f}%)</div>
                </div>
                <div class="section">
                    <h3 class="gainer">Maiores Altas</h3>
                    <ul>{gainers_html if gainers_html else "<li>Nenhuma alta significativa hoje</li>"}</ul>
                </div>
                <div class="section">
                    <h3 class="loser">Maiores Baixas</h3>
                    <ul>{losers_html if losers_html else "<li>Nenhuma queda significativa hoje</li>"}</ul>
                </div>
            </div>
        </div>
    </body>
    </html>
    """


def get_email_verification_html(name: str, verification_url: str) -> str:
    """Generate email verification HTML."""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
            .content {{ background: #f8f9fa; padding: 30px; border-radius: 0 0 10px 10px; }}
            .button {{ display: inline-block; background: #667eea; color: white; padding: 15px 30px; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
            .button:hover {{ background: #5a67d8; }}
            .info {{ background: #e3f2fd; border-left: 4px solid #2196f3; padding: 15px; margin: 20px 0; border-radius: 4px; }}
            .footer {{ text-align: center; padding: 20px; color: #666; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Confirme seu Email</h1>
            </div>
            <div class="content">
                <p>Olá, <strong>{name}</strong>!</p>
                <p>Obrigado por se cadastrar no InvestAI! Para completar seu registro e começar a usar todas as funcionalidades da plataforma, confirme seu endereço de email clicando no botão abaixo:</p>
                <p style="text-align: center;">
                    <a href="{verification_url}" class="button">Confirmar Email</a>
                </p>
                <div class="info">
                    <strong>Por que verificar seu email?</strong>
                    <ul>
                        <li>Garantir a segurança da sua conta</li>
                        <li>Receber alertas de preço e notificações</li>
                        <li>Recuperar sua senha se necessário</li>
                    </ul>
                </div>
                <p>Este link expira em 48 horas.</p>
                <p>Se você não criou uma conta no InvestAI, ignore este email.</p>
                <p style="word-break: break-all; background: #e5e7eb; padding: 10px; border-radius: 4px; font-size: 12px;">
                    Se o botão não funcionar, copie e cole este link no navegador:<br>{verification_url}
                </p>
            </div>
            <div class="footer">
                <p>Este é um email automático do InvestAI. Por favor, não responda.</p>
            </div>
        </div>
    </body>
    </html>
    """


def get_rebalance_suggestion_email_html(
    name: str,
    total_value: float,
    suggestions: list[dict],
    target_allocation: dict[str, float],
    current_allocation: dict[str, float],
) -> str:
    """Generate rebalance suggestion email HTML."""
    suggestions_html = ""
    for s in suggestions:
        action_color = "#059669" if s["action"] == "comprar" else "#dc2626"
        suggestions_html += f"""
        <tr>
            <td>{s['ticker']}</td>
            <td style="color: {action_color}; font-weight: bold;">{s['action'].upper()}</td>
            <td>R$ {s['amount']:,.2f}</td>
            <td>{s['current_pct']:.1f}% → {s['target_pct']:.1f}%</td>
        </tr>
        """

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }}
            .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 8px 8px; }}
            .summary {{ background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e5e7eb; }}
            th {{ background: #f3f4f6; font-weight: bold; }}
            .button {{ display: inline-block; background: #667eea; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; }}
            .info {{ background: #e0f2fe; border-left: 4px solid #0284c7; padding: 15px; margin: 20px 0; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>⚖️ Sugestão de Rebalanceamento</h1>
            </div>
            <div class="content">
                <p>Olá, <strong>{name}</strong>!</p>
                <p>Identificamos que sua carteira está desalinhada com sua alocação alvo. Aqui estão nossas sugestões de rebalanceamento:</p>

                <div class="summary">
                    <p><strong>Valor Total da Carteira:</strong> R$ {total_value:,.2f}</p>
                </div>

                <h3>Operações Sugeridas</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Ativo</th>
                            <th>Ação</th>
                            <th>Valor</th>
                            <th>Alocação</th>
                        </tr>
                    </thead>
                    <tbody>
                        {suggestions_html if suggestions_html else "<tr><td colspan='4'>Nenhuma operação necessária</td></tr>"}
                    </tbody>
                </table>

                <div class="info">
                    <strong>Por que rebalancear?</strong>
                    <p>O rebalanceamento ajuda a manter o risco do seu portfólio alinhado com seu perfil de investidor e aproveitar oportunidades de comprar na baixa e vender na alta.</p>
                </div>

                <p style="text-align: center;">
                    <a href="https://investai.com.br/portfolio/rebalance" class="button">Ver Detalhes no App</a>
                </p>

                <p style="font-size: 12px; color: #666; margin-top: 20px;">
                    <strong>Aviso:</strong> Esta é apenas uma sugestão baseada em sua alocação alvo. Considere seus objetivos pessoais e consulte um profissional antes de realizar operações.
                </p>
            </div>
        </div>
    </body>
    </html>
    """


def get_password_reset_email_html(name: str, reset_url: str) -> str:
    """Generate password reset email HTML."""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
            .content {{ background: #f8f9fa; padding: 30px; border-radius: 0 0 10px 10px; }}
            .button {{ display: inline-block; background: #667eea; color: white; padding: 15px 30px; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
            .button:hover {{ background: #5a67d8; }}
            .warning {{ background: #fef3c7; border-left: 4px solid #f59e0b; padding: 15px; margin: 20px 0; border-radius: 4px; }}
            .footer {{ text-align: center; padding: 20px; color: #666; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Redefinição de Senha</h1>
            </div>
            <div class="content">
                <p>Olá, <strong>{name}</strong>!</p>
                <p>Recebemos uma solicitação para redefinir sua senha no InvestAI.</p>
                <p>Clique no botão abaixo para criar uma nova senha:</p>
                <p style="text-align: center;">
                    <a href="{reset_url}" class="button">Redefinir Senha</a>
                </p>
                <div class="warning">
                    <strong>Importante:</strong>
                    <ul>
                        <li>Este link expira em 24 horas</li>
                        <li>Se você não solicitou esta redefinição, ignore este email</li>
                        <li>Nunca compartilhe este link com ninguém</li>
                    </ul>
                </div>
                <p>Se o botão não funcionar, copie e cole o link abaixo no seu navegador:</p>
                <p style="word-break: break-all; background: #e5e7eb; padding: 10px; border-radius: 4px; font-size: 12px;">{reset_url}</p>
            </div>
            <div class="footer">
                <p>Este é um email automático do InvestAI. Por favor, não responda.</p>
            </div>
        </div>
    </body>
    </html>
    """
