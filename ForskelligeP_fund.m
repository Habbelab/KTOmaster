data = readtable('TR indekser.xlsx', 'ReadVariableNames',true);
data = flipud(data);  % Ældste først

% Fjern rækker med NaN i relevante kolonner
relevanteKolonner = {'dlnp', 'x0HvisMeanReverse', 'regime_3', 'regime_5', 'regime_7'};
data = data(~any(ismissing(data(:, relevanteKolonner)), 2), :);

% Opret dummy-variable for regimer
regime0 = data.x0HvisMeanReverse == 0;
regime1 = data.x0HvisMeanReverse == 1;
% Opret lagget dLnP korrekt (efter vendt rækkefølge)
data.lag_dlnp = [NaN; data.dlnp(1:end-1)];
% Fjern første observation (lag = NaN)
data = data(2:end, :);
regime0 = regime0(2:end);
regime1 = regime1(2:end);

% Definér regressorer
lag_r0 = data.lag_dlnp .* regime0;
lag_r1 = data.lag_dlnp .* regime1;
cons_r1 = regime0;
cons_r2 = regime1;

% Saml X og y
X = [cons_r1, cons_r2, lag_r0, lag_r1];
y = data.dlnp;
dato = data.Dato;

% Estimér OLS
olsmdl = fitlm(X, y, "Intercept", false);

sigma2 = olsmdl.MSE;
cons1 = olsmdl.Coefficients.Estimate(1);
cons2 = olsmdl.Coefficients.Estimate(2);
AR1 = olsmdl.Coefficients.Estimate(3);
AR2 = olsmdl.Coefficients.Estimate(4);

MSmdl1_ini = arima('Constant', cons1, 'AR', AR1, 'Variance', sigma2);
MSmdl2_ini = arima('Constant', cons2, 'AR', AR2, 'Variance', sigma2);

% Konfiguration af modellen der skal estimeres (AR koefficienter = NaN,
% varians fast)
MSmdl1 = arima('Constant', NaN, 'AR', NaN, 'Variance', NaN);
MSmdl2 = arima('Constant', NaN, 'AR', NaN, 'Variance', NaN);
regimeNavne = ["Regime 1 - Mean reversion", "Regime 2 - Trend following"];
MSmdl3 = msVAR(dtmc([NaN NaN; NaN NaN], 'StateNames', regimeNavne), [MSmdl1, MSmdl2]);

% --- Grid af p11 og p22 ---
pvals1 = 0.51:0.02:0.59;
pvals2 = 0.11:0.02:0.19;

for i = 1:length(pvals1)
    p11 = pvals1(i);
    for j = 1:length(pvals2)
        p22 = pvals2(j);

        % Transition matrix
        P = [p11, 1-p11; 1-p22, p22];
        mc_ini = dtmc(P, 'StateNames', regimeNavne);
        MSmdl3_ini = msVAR(mc_ini, [MSmdl1_ini, MSmdl2_ini]);

        fprintf('\n==============================================\n');
        fprintf('Starter estimation for p11 = %.2f, p22 = %.2f\n', p11, p22);

        try
            EstMdl = estimate(MSmdl3, MSmdl3_ini, y, 'IterationPlot', false);
            summarize(EstMdl);
        catch ME
            fprintf('Estimation fejlede for p11 = %.2f, p22 = %.2f\n', p11, p22);
            fprintf('Fejl: %s\n', ME.message);
        end
    end
end